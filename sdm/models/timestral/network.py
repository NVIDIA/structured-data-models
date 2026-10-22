# Portions derived from mistralai/timestral (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor, nn

from sdm.models.timestral.config import _TimestralConfig
from sdm.models.timestral.icl import ICLBlock
from sdm.models.timestral.masks import (
    _build_forecast_positions,
    _build_forecast_time_mask,
    _build_two_stream_positions,
    _build_two_stream_time_mask,
    _build_two_stream_variate_mask,
    _compute_enc_key_validity,
)
from sdm.models.timestral.scaler import (
    _apply_affine_asinh_masked,
    _denormalize_quantiles,
    _instance_norm_stats,
)
from sdm.nn import CausalPatchNorm, MonotoneQuantiles, ResidualMLP

ROLE_TARGET = 0
ROLE_PAST_COV = 1
ROLE_KNOWN_COV = 2


class _PatchRepresentation(nn.Module):
    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = ResidualMLP(
            in_channels=2 * patch_size,
            hidden_channels=4 * d_model,
            out_channels=d_model,
            bias=True,
        )
        self.forecast_query = nn.Parameter(torch.zeros(d_model))

    def forward(self, scaled: Tensor, observed: Tensor) -> Tensor:
        r, t = scaled.shape
        n = t // self.patch_size
        values = scaled.reshape(r, n, self.patch_size)
        missing = (~observed).reshape(r, n, self.patch_size).to(scaled.dtype)
        x = torch.cat([values, missing], dim=-1).to(
            self.patch_proj.hidden_lin.weight.dtype
        )
        return self.patch_proj(x)

    def forecast_tokens(self, rows: int, n_patches: int) -> Tensor:
        return self.forecast_query.view(1, 1, -1).expand(rows, n_patches, -1)


class _QuantileHead(nn.Module):
    def __init__(self, config: _TimestralConfig) -> None:
        super().__init__()
        self.quantiles = MonotoneQuantiles(
            config.quantile_knots, config.quantile_width_factor
        )
        self.patch_size = config.patch_size
        self.num_quantiles = config.num_quantiles
        self.projection = ResidualMLP(
            in_channels=config.d_model,
            hidden_channels=4 * config.d_model,
            out_channels=config.patch_size * config.num_quantiles,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        raw = self.projection(x)
        *leading, n_patches, _ = raw.shape
        preds = raw.reshape(
            *leading, n_patches, self.patch_size, self.num_quantiles
        )
        preds = preds.to(torch.promote_types(preds.dtype, torch.float32))
        preds = self.quantiles(preds)
        return preds.reshape(
            *leading, n_patches * self.patch_size, self.num_quantiles
        )


class _TimestralModel(nn.Module):
    def __init__(self, config: _TimestralConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.norm = CausalPatchNorm(c.patch_size, c.minimum_scale, c.use_asinh)
        self.representation = _PatchRepresentation(c.patch_size, c.d_model)
        self.transformer = ICLBlock(
            channels=c.d_model,
            num_layers=len(c.block_schedule),
            column_every=c.column_every,
            num_heads=c.num_heads,
            hidden_channels=c.mlp_hidden_dim,
        )
        self.head = _QuantileHead(c)

    def scale(
        self, inputs: Tensor, observed: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.norm(inputs, observed)

    def _known_cov_scaling(
        self,
        inputs: Tensor,
        observed: Tensor,
        scaled: Tensor,
        loc: Tensor,
        scale: Tensor,
        known_rows: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        loc_c, scale_c, is_binary = _instance_norm_stats(
            values=inputs,
            observation_mask=observed,
            minimum_scale=self.config.minimum_scale,
        )
        known_scaled = _apply_affine_asinh_masked(
            data=inputs,
            loc=loc_c,
            scale=scale_c,
            observation_mask=observed,
            use_asinh=False,
        )
        if self.config.use_asinh:
            work = torch.promote_types(known_scaled.dtype, torch.float32)
            squashed = known_scaled.to(work).asinh().to(known_scaled.dtype)
            known_scaled = torch.where(
                observed & ~is_binary, squashed, known_scaled
            )
        sel = known_rows[:, None]
        return (
            torch.where(sel, known_scaled, scaled),
            torch.where(sel, loc_c, loc),
            torch.where(sel, scale_c, scale),
        )

    def _run_two_stream(
        self,
        x_enc: Tensor,
        x_dec: Tensor,
        time_mask: Tensor,
        variate_mask: Tensor | None,
        positions: Tensor,
    ) -> Tensor:
        x = torch.cat([x_enc, x_dec], dim=1)
        h = self.transformer(
            x=x,
            row_mask=time_mask,
            col_mask=variate_mask,
            pos=positions,
        )
        return self.head(h[:, x_enc.shape[1] :, :])

    def predict_block(
        self,
        inputs: Tensor,
        observed: Tensor,
        group_ids: Tensor,
        horizon_patches: int,
        num_variates: int,
        role_ids: Tensor | None = None,
        known_future: Tensor | None = None,
        known_future_mask: Tensor | None = None,
    ) -> Tensor:
        if role_ids is None or not self.config.covariate_enabled:
            return self._predict_context_only(
                inputs=inputs,
                observed=observed,
                group_ids=group_ids,
                horizon_patches=horizon_patches,
                num_variates=num_variates,
            )
        r = inputs.shape[0]
        horizon_len = horizon_patches * self.config.patch_size
        if known_future is None:
            known_future = torch.zeros(
                r, horizon_len, device=inputs.device, dtype=inputs.dtype
            )
            known_future_mask = torch.zeros(
                r, horizon_len, device=inputs.device, dtype=torch.bool
            )
        assert known_future_mask is not None
        full_inputs = torch.cat([inputs, known_future], dim=-1)
        full_observed = torch.cat([observed, known_future_mask], dim=-1)
        return self._predict_window(
            full_inputs=full_inputs,
            full_observed=full_observed,
            group_ids=group_ids,
            horizon_patches=horizon_patches,
            num_variates=num_variates,
            role_ids=role_ids,
        )

    def _predict_context_only(
        self,
        inputs: Tensor,
        observed: Tensor,
        group_ids: Tensor,
        horizon_patches: int,
        num_variates: int,
    ) -> Tensor:
        c = self.config
        ps = c.patch_size
        rows, t = inputs.shape
        ll, h = (t // ps, horizon_patches)
        scaled, loc, scale = self.scale(inputs, observed)
        x_enc = self.representation(scaled, observed)
        x_dec = self.representation.forecast_tokens(rows, h)
        obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
        key_valid = _compute_enc_key_validity(
            obs_per_patch=obs_per_patch,
            attend_missing=c.attend_missing_patches,
        )
        time_mask = _build_forecast_time_mask(ll, h, key_valid, inputs.device)
        variate_mask = _build_two_stream_variate_mask(group_ids, key_valid, h)
        # Packed rows are contiguous [batch, variate, patch].
        first_obs = obs_per_patch.unflatten(0, (-1, num_variates)).any(1)
        first_obs = first_obs.int().argmax(-1).repeat_interleave(num_variates)
        positions = _build_forecast_positions(ll, h, first_obs)
        preds = self._run_two_stream(
            x_enc=x_enc,
            x_dec=x_dec,
            time_mask=time_mask,
            variate_mask=variate_mask,
            positions=positions,
        )
        anchor_loc = loc.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
        return _denormalize_quantiles(
            preds=preds,
            loc=anchor_loc.unsqueeze(-1),
            scale=anchor_scale.unsqueeze(-1),
            use_asinh=c.use_asinh,
        )

    def _predict_window(
        self,
        full_inputs: Tensor,
        full_observed: Tensor,
        group_ids: Tensor,
        horizon_patches: int,
        num_variates: int,
        role_ids: Tensor,
    ) -> Tensor:
        c = self.config
        ps = c.patch_size
        rows, w = full_inputs.shape
        h = horizon_patches
        n = w // ps
        ll = n - h
        target_rows = role_ids == ROLE_TARGET
        known_rows = role_ids == ROLE_KNOWN_COV
        scaled, loc, scale = self.scale(full_inputs, full_observed)
        scaled, loc, scale = self._known_cov_scaling(
            inputs=full_inputs,
            observed=full_observed,
            scaled=scaled,
            loc=loc,
            scale=scale,
            known_rows=known_rows,
        )
        x_enc = self.representation(scaled, full_observed)
        x_dec = self.representation.forecast_tokens(rows, n)
        obs_per_patch = full_observed.unflatten(-1, (n, ps)).any(dim=-1)
        key_valid = _compute_enc_key_validity(
            obs_per_patch=obs_per_patch,
            attend_missing=c.attend_missing_patches,
        )
        s_rows = torch.full(
            (rows,), ll, device=full_inputs.device, dtype=torch.long
        )
        time_mask = _build_two_stream_time_mask(
            min_context=s_rows,
            n_patches=n,
            block_size=c.block_size,
            key_valid=key_valid,
            bidirectional_rows=known_rows,
        )
        variate_mask = _build_two_stream_variate_mask(
            group_ids=group_ids,
            enc_key_valid=key_valid,
            n_dec=n,
            target_rows=target_rows,
            bidirectional_rows=known_rows,
        )
        # Packed rows are contiguous [batch, variate, patch].
        first_obs = obs_per_patch.unflatten(0, (-1, num_variates)).any(1)
        first_obs = first_obs.int().argmax(-1).repeat_interleave(num_variates)
        positions = _build_two_stream_positions(n, first_obs)
        preds = self._run_two_stream(
            x_enc=x_enc,
            x_dec=x_dec,
            time_mask=time_mask,
            variate_mask=variate_mask,
            positions=positions,
        ).unflatten(1, (n, ps))
        horizon_preds = preds[:, ll:n].reshape(rows, h * ps, preds.shape[-1])
        anchor_loc = loc.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.unflatten(-1, (n, ps))[..., 0][:, ll - 1 : ll]
        return _denormalize_quantiles(
            preds=horizon_preds,
            loc=anchor_loc.unsqueeze(-1),
            scale=anchor_scale.unsqueeze(-1),
            use_asinh=c.use_asinh,
        )
