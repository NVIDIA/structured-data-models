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
