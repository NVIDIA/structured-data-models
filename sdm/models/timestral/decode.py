# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm.models.timestral.network import (
    ROLE_KNOWN_COV,
    ROLE_PAST_COV,
    ROLE_TARGET,
    _TimestralModel,
)
from sdm.models.timestral.scaler import (
    _advancing_block_frames,
    _clamp_nonfinite,
)


def _ffill_with_mask(inputs: Tensor, mask: Tensor) -> Tensor:
    idx = torch.arange(inputs.shape[-1], device=inputs.device).expand_as(
        inputs
    )
    idx = torch.where(mask, idx, torch.zeros_like(idx))
    idx = idx.cummax(dim=-1).values
    return torch.nan_to_num(torch.gather(inputs, dim=-1, index=idx), nan=0.0)


def _pad_to_patch(
    inputs: Tensor, mask: Tensor, patch_size: int
) -> tuple[Tensor, Tensor]:
    ctx_len = inputs.shape[-1]
    if ctx_len % patch_size != 0:
        pad_len = patch_size - ctx_len % patch_size
        inputs = torch.nn.functional.pad(inputs, (pad_len, 0), value=0.0)
        mask = torch.nn.functional.pad(mask, (pad_len, 0), value=False)
    return (inputs, mask)


def _replace_extreme_values(
    x: Tensor, threshold: float, replacement: float = 0.0
) -> Tensor:
    return torch.where(
        torch.isfinite(x) & (x.abs() < threshold),
        x,
        torch.full_like(x, replacement),
    )


class _Decoder:
    def __init__(
        self,
        model: _TimestralModel,
        max_context: int | None = None,
        per_shot_horizon_patches: int = 32,
        quantile_real_cap_k: float = 10000.0,
        extreme_value_threshold: float = 1e30,
    ) -> None:
        self.model = model
        self.config = model.config
        self.max_context = (
            max_context
            if max_context is not None
            else model.config.context_length
        )
        self.per_shot_horizon_patches = min(
            per_shot_horizon_patches, model.config.block_size
        )
        self.quantile_real_cap_k = quantile_real_cap_k
        self.extreme_value_threshold = extreme_value_threshold
        self.median_index = model.config.median_index

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    def _forecast_raw(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        past_covariates: Tensor | None = None,
        known_covariates: Tensor | None = None,
    ) -> Tensor:
        rows, mask, role_ids, known_future, known_future_mask, n_tgt = (
            self._assemble_rows(
                inputs=inputs,
                observation_mask=observation_mask,
                prediction_length=prediction_length,
                past_covariates=past_covariates,
                known_covariates=known_covariates,
            )
        )
        quantiles = self._block_ar_decode(
            inputs=rows,
            observation_mask=mask,
            prediction_length=prediction_length,
            role_ids=role_ids,
            known_future=known_future,
            known_future_mask=known_future_mask,
        )
        quantiles = quantiles[:, :n_tgt]
        if self.quantile_real_cap_k > 0:
            quantiles = self._cap_quantiles(
                quantiles=quantiles,
                inputs=rows[:, :n_tgt],
                observation_mask=mask[:, :n_tgt],
            )
        return quantiles

    def _cap_quantiles(
        self, quantiles: Tensor, inputs: Tensor, observation_mask: Tensor
    ) -> Tensor:
        not_obs = ~observation_mask
        ctx_max = torch.nan_to_num(
            inputs.masked_fill(not_obs, float("-inf")).amax(-1, keepdim=True),
            neginf=0.0,
        )
        ctx_min = torch.nan_to_num(
            inputs.masked_fill(not_obs, float("inf")).amin(-1, keepdim=True),
            posinf=0.0,
        )
        _, scale = self.model.norm.stats(inputs, observation_mask)
        anchor_scale = scale[..., -1:] * self.quantile_real_cap_k
        return quantiles.clamp(
            (ctx_min - anchor_scale).unsqueeze(-1),
            (ctx_max + anchor_scale).unsqueeze(-1),
        )

    def _assemble_rows(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        past_covariates: Tensor | None,
        known_covariates: Tensor | None,
    ) -> tuple[
        Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, int
    ]:
        ps = self.patch_size
        device = inputs.device
        b, n_tgt, _ctx0 = inputs.shape
        horizon = math.ceil(prediction_length / ps) * ps
        ctx_rows = [_ffill_with_mask(inputs, observation_mask)]
        ctx_masks = [observation_mask]
        roles = [ROLE_TARGET] * n_tgt
        n_past = 0
        if past_covariates is not None:
            n_past = past_covariates.shape[1]
            ctx_masks.append(~torch.isnan(past_covariates))
            ctx_rows.append(torch.nan_to_num(past_covariates, nan=0.0))
            roles += [ROLE_PAST_COV] * n_past
        n_known = 0
        known_future_vals: Tensor | None = None
        known_future_obs: Tensor | None = None
        if known_covariates is not None:
            n_known = known_covariates.shape[1]
            kc_ctx, kc_fut = (
                known_covariates[..., :-prediction_length],
                known_covariates[..., -prediction_length:],
            )
            ctx_masks.append(~torch.isnan(kc_ctx))
            ctx_rows.append(torch.nan_to_num(kc_ctx, nan=0.0))
            roles += [ROLE_KNOWN_COV] * n_known
            known_future_obs = ~torch.isnan(kc_fut)
            known_future_vals = torch.nan_to_num(kc_fut, nan=0.0)
        rows = torch.cat(ctx_rows, dim=1)
        mask = torch.cat(ctx_masks, dim=1)
        if rows.shape[-1] > self.max_context:
            rows = rows[:, :, -self.max_context :]
            mask = mask[:, :, -self.max_context :]
        rows, mask = _pad_to_patch(rows, mask, ps)
        v_all = rows.shape[1]
        if n_past == 0 and n_known == 0:
            return (rows, mask, None, None, None, n_tgt)
        role_ids = torch.tensor(roles, dtype=torch.long, device=device).repeat(
            b
        )
        known_future: Tensor | None = None
        known_future_mask: Tensor | None = None
        if known_future_vals is not None and known_future_obs is not None:
            k0 = n_tgt + n_past
            plen = known_future_vals.shape[-1]
            kf = torch.zeros(
                b, v_all, horizon, device=device, dtype=rows.dtype
            )
            kfm = torch.zeros(
                b, v_all, horizon, device=device, dtype=torch.bool
            )
            kf[:, k0 : k0 + n_known, :plen] = known_future_vals
            kfm[:, k0 : k0 + n_known, :plen] = known_future_obs
            known_future = kf.reshape(b * v_all, horizon)
            known_future_mask = kfm.reshape(b * v_all, horizon)
        return (rows, mask, role_ids, known_future, known_future_mask, n_tgt)

    def _block_ar_decode(
        self,
        inputs: Tensor,
        observation_mask: Tensor,
        prediction_length: int,
        role_ids: Tensor | None,
        known_future: Tensor | None,
        known_future_mask: Tensor | None,
    ) -> Tensor:
        ps = self.patch_size
        b, v, ctx_len = inputs.shape
        context = inputs.reshape(b * v, ctx_len)
        mask = observation_mask.reshape(b * v, ctx_len)
        group_ids = torch.arange(b, device=inputs.device).repeat_interleave(v)
        known_rows = past_rows = target_rows = None
        if role_ids is not None:
            known_rows = role_ids == ROLE_KNOWN_COV
            past_rows = (role_ids != ROLE_TARGET) & ~known_rows
            target_rows = role_ids == ROLE_TARGET
        blocks: list[Tensor] = []
        remaining = math.ceil(prediction_length / ps)
        offset = 0
        while remaining > 0:
            block = min(self.per_shot_horizon_patches, remaining)
            blen = block * ps
            kf_block = (
                known_future[:, offset : offset + blen]
                if known_future is not None
                else None
            )
            kfm_block = (
                known_future_mask[:, offset : offset + blen]
                if known_future_mask is not None
                else None
            )
            quantiles = self.model.predict_block(
                inputs=context,
                observed=mask,
                group_ids=group_ids,
                horizon_patches=block,
                num_variates=v,
                role_ids=role_ids,
                known_future=kf_block,
                known_future_mask=kfm_block,
            )
            quantiles = self._refine_block(
                quantiles=quantiles,
                context=context,
                mask=mask,
                target_rows=target_rows,
            )
            blocks.append(quantiles)
            remaining -= block
            offset += blen
            if remaining <= 0:
                break
            feedback = _replace_extreme_values(
                x=quantiles[..., self.median_index],
                threshold=self.extreme_value_threshold,
            )
            feedback_mask = torch.ones_like(feedback, dtype=torch.bool)
            if role_ids is not None:
                assert known_rows is not None
                assert past_rows is not None
                if kf_block is not None:
                    assert kfm_block is not None
                    feedback = torch.where(
                        known_rows[:, None], kf_block, feedback
                    )
                    feedback_mask = torch.where(
                        known_rows[:, None], kfm_block, feedback_mask
                    )
                feedback = torch.where(
                    past_rows[:, None], torch.zeros_like(feedback), feedback
                )
                feedback_mask = feedback_mask & ~past_rows[:, None]
            context = torch.cat([context, feedback], dim=-1)
            mask = torch.cat([mask, feedback_mask], dim=-1)
            if context.shape[-1] > self.max_context:
                keep = max(ps, self.max_context // ps * ps)
                context = context[..., -keep:]
                mask = mask[..., -keep:]
        packed = torch.cat(blocks, dim=-2)[..., :prediction_length, :]
        return packed.reshape(b, v, prediction_length, -1)

    def _refine_block(
        self,
        quantiles: Tensor,
        context: Tensor,
        mask: Tensor,
        target_rows: Tensor | None,
    ) -> Tensor:
        ps = self.patch_size
        n_pred = quantiles.shape[1] // ps
        if n_pred <= 1:
            return quantiles
        _, loc, scale = self.model.scale(context, mask)
        loc0, scale0 = (loc[:, -1], scale[:, -1])
        median = (
            quantiles[..., self.median_index]
            .unflatten(-1, (n_pred, ps))
            .to(torch.float64)
        )
        sinh_median = (median - loc0[:, None, None]) / scale0[:, None, None]
        loc_j, scale_j = _advancing_block_frames(
            context=context,
            observation_mask=mask,
            sinh_median=sinh_median,
            minimum_scale=self.config.minimum_scale,
        )
        first = torch.zeros(loc_j.shape, dtype=torch.bool, device=loc_j.device)
        first[:, 0] = True
        loc_j = torch.where(
            first, loc0[:, None], loc_j.to(quantiles.dtype)
        ).repeat_interleave(ps, dim=-1)
        scale_j = torch.where(
            first, scale0[:, None], scale_j.to(quantiles.dtype)
        ).repeat_interleave(ps, dim=-1)
        sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
        refined = _clamp_nonfinite(
            vals=sinh_norm * scale_j[..., None] + loc_j[..., None]
        )
        if target_rows is not None:
            refined = torch.where(
                target_rows[:, None, None], refined, quantiles
            )
        return refined
