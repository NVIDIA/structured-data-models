# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import Tensor

from sdm.models.timesfm3.configs import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.cpm_revin_refine import (
    cpm_iterative_revin_refine,
)
from sdm.models.timesfm3.dense import ResidualBlock
from sdm.models.timesfm3.transformer import StackedMixingTransformer
from sdm.models.timesfm3.util import (
    get_output_patch_via_roll,
    get_running_stats,
    revin,
)


class _TimesFM3Model(torch.nn.Module):
    """Implement the Google TimesFM-3 full-sequence model.

    Args:
        input_patch_len: Number of time steps in each input patch.
        output_patch_len: Number of time steps predicted from each patch.
        quantiles: Quantiles predicted by the output head.
        residual_block_config: Pre-transformer residual-block configuration.
        transformer_config: Transformer-stack configuration.
        use_variate_attention: Whether to attend across variates.
        value_clip: Absolute bound applied to inputs and predictions.
        use_stitching: Whether decoding will stitch overlapping predictions.
        use_linear_detrending: Whether decoding will remove linear trends.
        linear_detrending_threshold: Ratio controlling linear detrending.
        use_iterative_cpm_revin: Whether to refine RevIN statistics for
            Contiguous Patch Masking positions.
        use_frozen_running_stats: Whether decoding will freeze statistics at
            the context boundary.
        device: Device on which to create parameters and buffers.
        dtype: Data type of parameters.
    """

    def __init__(
        self,
        input_patch_len: int = 32,
        output_patch_len: int = 64,
        quantiles: Sequence[float] | None = None,
        residual_block_config: ResidualBlockConfig
        | dict[str, Any]
        | None = None,
        transformer_config: StackedTransformersConfig
        | dict[str, Any]
        | None = None,
        use_variate_attention: bool = True,
        value_clip: float = 1e20,
        use_stitching: bool = True,
        use_linear_detrending: bool = True,
        linear_detrending_threshold: float = 0.5,
        use_iterative_cpm_revin: bool = True,
        use_frozen_running_stats: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if quantiles is None:
            quantiles = [index / 10 for index in range(1, 10)]
        if residual_block_config is None:
            residual_block_config = ResidualBlockConfig(
                hidden_dims=1280,
                output_dims=1280,
                use_bias=False,
                activation="relu",
            )
        elif isinstance(residual_block_config, dict):
            residual_block_config = ResidualBlockConfig(
                **cast(dict[str, Any], residual_block_config)
            )

        if transformer_config is None:
            transformer_config = StackedTransformersConfig(
                num_layers=20,
                transformer=TransformerConfig(
                    model_dims=1280,
                    hidden_dims=1280,
                    num_heads=16,
                    qk_norm="rms",
                    use_rope_seq=True,
                    use_rope_var=False,
                    use_bias=False,
                    ff_activation="relu",
                ),
            )
        elif isinstance(transformer_config, dict):
            transformer_data = transformer_config.get("transformer", {})
            if isinstance(transformer_data, dict):
                transformer_data = TransformerConfig(
                    **cast(dict[str, Any], transformer_data)
                )
            stack_data = dict(transformer_config)
            stack_data["transformer"] = transformer_data
            transformer_config = StackedTransformersConfig(**stack_data)

        if output_patch_len % input_patch_len != 0:
            raise ValueError(
                f"Output patch length {output_patch_len} must be a multiple "
                f"of input patch length {input_patch_len}."
            )
        if (
            residual_block_config.output_dims
            != transformer_config.transformer.model_dims
        ):
            raise ValueError(
                "Residual-block output dimensions must match transformer "
                "model dimensions."
            )
        if use_stitching and output_patch_len <= input_patch_len:
            raise ValueError(
                "Stitching requires output_patch_len > input_patch_len."
            )

        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.quantiles = list(quantiles)
        self.num_quantiles = len(self.quantiles)
        self.rolls = output_patch_len // input_patch_len
        self.residual_block_config = residual_block_config
        self.transformer_config = transformer_config
        self.use_variate_attention = use_variate_attention
        self.value_clip = value_clip
        self.use_stitching = use_stitching
        self.use_linear_detrending = use_linear_detrending
        self.linear_detrending_threshold = linear_detrending_threshold
        self.use_iterative_cpm_revin = use_iterative_cpm_revin
        self.use_frozen_running_stats = use_frozen_running_stats

        if use_stitching:
            self._stitching_extract_len = min(
                2 * input_patch_len,
                output_patch_len,
            )

        self.pre_transformer_resblock = ResidualBlock(
            config=residual_block_config,
            input_dims=2 * (input_patch_len + output_patch_len),
            device=device,
            dtype=dtype,
        )
        self.transformer_stack = StackedMixingTransformer(
            config=transformer_config,
            use_variate_attention=use_variate_attention,
            device=device,
            dtype=dtype,
        )
        self.output_head = torch.nn.Linear(
            transformer_config.transformer.model_dims,
            output_patch_len * self.num_quantiles,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def _preprocess(
        self,
        values: Tensor,
        masks: Tensor,
        patch_is_target: Tensor,
        freeze_after: int | None = None,
        patch_cpm_mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        tuple[Tensor, Tensor],
        Tensor,
    ]:
        running_n, running_mean, running_std = get_running_stats(
            values,
            masks,
        )
        if freeze_after is not None:
            num_patches = values.shape[2]
            if 0 <= freeze_after < num_patches - 1:
                running_mean[:, :, freeze_after + 1 :] = running_mean[
                    :, :, freeze_after : freeze_after + 1
                ]
                running_std[:, :, freeze_after + 1 :] = running_std[
                    :, :, freeze_after : freeze_after + 1
                ]

        if patch_cpm_mask is not None:
            cpm_mask = patch_cpm_mask[:, None, :, None]
            masks = masks | (cpm_mask & patch_is_target.unsqueeze(-1))

        normalized_values = revin(
            values,
            running_mean,
            running_std,
        )
        normalized_values = torch.where(masks, 0.0, normalized_values)

        future_values, wrap_mask = get_output_patch_via_roll(
            values,
            self.rolls,
        )
        future_values = revin(
            future_values,
            running_mean,
            running_std,
        )
        future_masks, _ = get_output_patch_via_roll(masks, self.rolls)
        future_masks = future_masks | patch_is_target.unsqueeze(-1) | wrap_mask
        future_values = torch.where(future_masks, 0.0, future_values)

        values_with_future = torch.cat(
            [normalized_values, future_values],
            dim=-1,
        )
        masks_with_future = torch.cat([masks, future_masks], dim=-1)
        input_dtype = self.pre_transformer_resblock.hidden_layer.weight.dtype
        residual_input = torch.cat(
            [
                values_with_future.to(input_dtype),
                masks_with_future.to(input_dtype),
            ],
            dim=-1,
        )
        transformer_input = self.pre_transformer_resblock(residual_input)
        patch_mask = masks_with_future.all(dim=-1)

        return (
            residual_input,
            transformer_input,
            patch_mask,
            (running_mean, running_std),
            running_n,
        )

    def forward(
        self,
        values: Tensor,
        masks: Tensor,
        patch_is_target: Tensor,
        *,
        freeze_after: int | None = None,
        patch_cpm_mask: Tensor | None = None,
        return_aux_outputs: bool = False,
    ) -> dict[str, Any]:
        """Predict every quantile for each input patch.

        Args:
            values: Patched series with shape ``[B, V, N, P]``.
            masks: Invalid-value mask with shape ``[B, V, N, P]``.
            patch_is_target: Target-patch indicator with shape ``[B, V, N]``.
            freeze_after: Optional final patch included in running statistics.
            patch_cpm_mask: Contiguous Patch Masking indicator with shape
                ``[B, N]``.
            return_aux_outputs: Whether to include intermediate tensors.

        Returns:
            Mapping containing ``logits`` with shape ``[B, V, N, O, Q]`` and
            the RevIN statistics used for denormalization.
        """
        values = values.nan_to_num(nan=0.0).clamp(
            -self.value_clip,
            self.value_clip,
        )
        masks = masks.bool()
        if values.shape[-1] != self.input_patch_len:
            raise ValueError(
                f"Input patch length {values.shape[-1]} does not match "
                f"configured length {self.input_patch_len}."
            )

        (
            residual_input,
            transformer_input,
            transformer_patch_mask,
            revin_stats,
            running_n,
        ) = self._preprocess(
            values,
            masks,
            patch_is_target,
            freeze_after=freeze_after,
            patch_cpm_mask=patch_cpm_mask,
        )

        effective_patch_mask = transformer_patch_mask.cummin(dim=2).values
        transformer_output, attention_masks = self.transformer_stack(
            transformer_input,
            effective_patch_mask,
        )
        raw_logits = self.output_head(transformer_output)
        revin_mean, revin_std = revin_stats

        if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
            refined_mean, refined_std = cpm_iterative_revin_refine(
                raw_logits,
                revin_n=running_n,
                revin_mu=revin_mean,
                revin_sigma=revin_std,
                patch_cpm_mask=patch_cpm_mask,
                median_q_idx=self.num_quantiles // 2,
                rolls=self.rolls,
                patch_len=self.input_patch_len,
                num_quantiles=self.num_quantiles,
                value_clip=self.value_clip,
            )
            cpm_mask = patch_cpm_mask.unsqueeze(1)
            revin_mean = torch.where(cpm_mask, refined_mean, revin_mean)
            revin_std = torch.where(cpm_mask, refined_std, revin_std)

        logits = revin(
            raw_logits,
            revin_mean,
            revin_std,
            reverse=True,
        ).clamp(-self.value_clip, self.value_clip)
        batch_size, num_variates, num_patches = logits.shape[:3]
        logits = logits.view(
            batch_size,
            num_variates,
            num_patches,
            self.output_patch_len,
            self.num_quantiles,
        )

        outputs: dict[str, Any] = {
            "logits": logits,
            "revin_stats": revin_stats,
        }
        if return_aux_outputs:
            outputs["__call__:resblock_input"] = residual_input
            outputs["__call__:transformer_input"] = transformer_input
            outputs["__call__:seq_attn_mask"] = attention_masks
            outputs["__call__:transformer_output"] = transformer_output
        return outputs

    def decode(
        self,
        target: Tensor,
        horizon: int = 0,
        past_only_covariates: Tensor | None = None,
        past_future_covariates: Tensor | None = None,
        target_mask: Tensor | None = None,
        past_only_mask: Tensor | None = None,
        past_future_mask: Tensor | None = None,
        mask: Tensor | None = None,
        return_aux_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        """Decode a forecast in one non-autoregressive pass.

        Future-known covariates determine the forecast horizon when supplied,
        overriding ``horizon``.

        Gradient tracking follows the ambient PyTorch gradient mode.

        Args:
            target: Target context with shape ``[B, U, C]``, where ``B`` is the
                batch size and ``U`` is the number of target variates.
                ``C`` is the context length.
            horizon: Number of future time steps to predict.
            past_only_covariates: Historical covariates with shape
                ``[B, V, C]``.
            past_future_covariates: Future-known covariates with shape
                ``[B, W, C + H]``, where ``H`` is the forecast horizon.
            target_mask: Invalid-value mask with shape ``[B, U, C]``.
            past_only_mask: Invalid-value mask with shape ``[B, V, C]``.
            past_future_mask: Invalid-value mask with shape
                ``[B, W, C + H]``.
            mask: Global context mask with shape ``[B, C]``.
            return_aux_outputs: Whether to return full-sequence intermediate
                outputs with the forecast.

        Returns:
            Forecasts for every target and covariate variate with shape
            ``[B, U + V + W, H, Q]``, where ``Q`` is the number of quantiles,
            optionally paired with the full-sequence outputs.
        """
        device = target.device
        batch_size, num_target, context = target.shape

        if past_future_covariates is not None:
            horizon = past_future_covariates.shape[-1] - context
        if horizon <= 0:
            raise ValueError("Decode function requires horizon > 0.")
        if self.use_stitching or self.use_linear_detrending:
            raise NotImplementedError(
                "Stitched decoding and linear detrending are not yet "
                "implemented. Set both options to False."
            )

        # 1. Pad context to multiple of input_patch_len
        ctx_padding = (
            self.input_patch_len - (context % self.input_patch_len)
        ) % self.input_patch_len
        if ctx_padding > 0:
            target = torch.nn.functional.pad(target, (ctx_padding, 0))
            if mask is not None:
                mask = torch.nn.functional.pad(
                    mask, (ctx_padding, 0), value=True
                )
            if past_only_covariates is not None:
                past_only_covariates = torch.nn.functional.pad(
                    past_only_covariates, (ctx_padding, 0)
                )
            if past_future_covariates is not None:
                past_future_covariates = torch.nn.functional.pad(
                    past_future_covariates, (ctx_padding, 0)
                )
            if target_mask is not None:
                target_mask = torch.nn.functional.pad(
                    target_mask, (ctx_padding, 0), value=True
                )
            if past_only_mask is not None:
                past_only_mask = torch.nn.functional.pad(
                    past_only_mask, (ctx_padding, 0), value=True
                )
            if past_future_mask is not None:
                past_future_mask = torch.nn.functional.pad(
                    past_future_mask, (ctx_padding, 0), value=True
                )
            context = context + ctx_padding

        if mask is None:
            mask = torch.zeros(
                batch_size, context, dtype=torch.bool, device=device
            )
            if ctx_padding > 0:
                mask[:, :ctx_padding] = True

        # 2. Pad horizon
        hor_padding = (-horizon) % self.output_patch_len
        padded_horizon = horizon + hor_padding
        num_horizon_patches = padded_horizon // self.input_patch_len
        num_context_patches = context // self.input_patch_len

        # 3. Build context & horizon inputs
        if target_mask is None:
            target_mask = torch.zeros_like(target, dtype=torch.bool)
        target_mask = target_mask | mask.unsqueeze(1)

        all_ctx_vals = [target]
        all_ctx_masks = [target_mask]
        num_past_only = 0
        if past_only_covariates is not None:
            num_past_only = past_only_covariates.shape[1]
            if past_only_mask is None:
                past_only_mask = torch.zeros_like(
                    past_only_covariates, dtype=torch.bool
                )
            all_ctx_vals.append(past_only_covariates)
            all_ctx_masks.append(past_only_mask | mask.unsqueeze(1))
        if past_future_covariates is not None:
            if past_future_mask is None:
                past_future_mask = torch.zeros_like(
                    past_future_covariates, dtype=torch.bool
                )
            all_ctx_vals.append(past_future_covariates[..., :context])
            all_ctx_masks.append(
                past_future_mask[..., :context] | mask.unsqueeze(1)
            )

        ctx_vals = torch.cat(all_ctx_vals, dim=1)
        ctx_masks = torch.cat(all_ctx_masks, dim=1)

        ctx_vals = torch.where(ctx_masks, 0.0, ctx_vals)

        all_hor_vals = [
            torch.zeros(batch_size, num_target, padded_horizon, device=device),
            torch.zeros(
                batch_size, num_past_only, padded_horizon, device=device
            ),
        ]
        all_hor_masks = [
            torch.ones(
                batch_size,
                num_target,
                padded_horizon,
                dtype=torch.bool,
                device=device,
            ),
            torch.ones(
                batch_size,
                num_past_only,
                padded_horizon,
                dtype=torch.bool,
                device=device,
            ),
        ]

        if past_future_covariates is not None:
            if past_future_mask is None:
                past_future_mask = torch.zeros_like(
                    past_future_covariates, dtype=torch.bool
                )
            pf_future_vals = past_future_covariates[
                ..., context : context + horizon
            ]
            pf_future_masks = past_future_mask[
                ..., context : context + horizon
            ]
            pf_future_vals = torch.where(pf_future_masks, 0.0, pf_future_vals)
            if hor_padding > 0:
                pf_future_vals = torch.nn.functional.pad(
                    pf_future_vals, (0, hor_padding)
                )
                pf_future_masks = torch.nn.functional.pad(
                    pf_future_masks, (0, hor_padding), value=True
                )
            all_hor_vals.append(pf_future_vals)
            all_hor_masks.append(pf_future_masks)

        hor_vals = torch.cat(all_hor_vals, dim=1)
        hor_masks = torch.cat(all_hor_masks, dim=1)

        all_vals = torch.cat([ctx_vals, hor_vals], dim=-1)
        all_masks = torch.cat([ctx_masks, hor_masks], dim=-1)

        num_variates = all_vals.shape[1]
        patch_is_target = torch.zeros(
            (
                batch_size,
                num_variates,
                num_context_patches + num_horizon_patches,
            ),
            dtype=torch.bool,
            device=device,
        )
        patch_is_target[:, : num_target + num_past_only, :] = True

        # Reshape values & masks to patched shape (b, v, n, p)
        values_bvnp = all_vals.reshape(
            batch_size, num_variates, -1, self.input_patch_len
        )
        masks_bvnp = all_masks.reshape(
            batch_size, num_variates, -1, self.input_patch_len
        )

        # Build horizon CPM mask: context=False, horizon=True.
        num_total_patches = num_context_patches + num_horizon_patches
        horizon_cpm_mask = torch.zeros(
            batch_size, num_total_patches, dtype=torch.bool, device=device
        )
        horizon_cpm_mask[:, num_context_patches:] = True

        freeze_after = (
            num_context_patches - 1 if self.use_frozen_running_stats else None
        )
        forward_out = self.forward(
            values_bvnp,
            masks_bvnp,
            patch_is_target,
            freeze_after=freeze_after,
            patch_cpm_mask=horizon_cpm_mask,
            return_aux_outputs=return_aux_outputs,
        )
        logits = forward_out[
            "logits"
        ]  # (b, v, n, output_patch_len, num_quantiles)

        num_forecast_chunks = padded_horizon // self.output_patch_len
        forecast_indices = torch.arange(
            num_forecast_chunks, device=device
        ) * self.rolls + (num_context_patches - 1)
        forecast_logits = logits[:, :, forecast_indices, :, :]
        horizon_logits = forecast_logits.reshape(
            batch_size, num_variates, -1, self.num_quantiles
        )[:, :, :horizon, :]

        if return_aux_outputs:
            return horizon_logits, forward_out
        return horizon_logits
