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

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from itertools import product
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, StypeLike, TableTensor, Task
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.timesfm3.configs import (
    ResidualBlockConfig,
    StackedTransformersConfig,
    TransformerConfig,
)
from sdm.models.timesfm3.cpm_revin_refine import (
    cpm_iterative_revin_refine,
)
from sdm.models.timesfm3.dense import ResidualBlock
from sdm.models.timesfm3.recipe import default_recipe
from sdm.models.timesfm3.transformer import StackedMixingTransformer
from sdm.models.timesfm3.util import (
    get_output_patch_via_roll,
    get_running_stats,
    revin,
)
from sdm.tensor.table import TableSchema


class TimesFM3(ICLModel):
    r"""The multivariate forecasting foundation model from `"TimesFM-3: A
    Zero-shot Foundation Model for Multivariate Forecasting"
    <https://research.google/blog/
    timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting>`__.

    .. figure:: /images/timesfm3_light.png
        :figclass: light-only
        :width: 100%

    .. figure:: /images/timesfm3_dark.png
        :figclass: dark-only
        :width: 100%

    :class:`TimesFM3` is a zero-shot time-series foundation model for
    multivariate forecasting. It extends earlier univariate TimesFM models with
    native support for jointly forecasting multiple coevolving target series,
    incorporating historical covariates, and using dynamic covariates that are
    known across both the past and future forecast horizon.

    Architecturally, it combines patch-based time-series tokenization with
    alternating causal temporal attention and full variate attention, allowing
    forecasts to use both within-series history and cross-series dependencies.
    It decodes the full forecast horizon in a single forward pass and returns
    nine quantile forecasts, from the 10th to the 90th percentile, for each
    target series and query time step.

    Within the :class:`~sdm.models.ICLModel` protocol, :class:`TimesFM3`
    treats rows as ordered time steps. In a default forward pass,
    ``x_context`` contains historical covariates, ``y_context`` contains one or
    more past target series, and ``x_query`` contains future-known covariates
    (which must also be present in ``x_context``).

    .. note::
        :class:`TimesFM` model weights are distributed under the
        `TimesFM Non-Commercial License v1.0 <https://huggingface.co/google/
        timesfm-3.0-pytorch/blob/main/LICENSE>`__.
        Before downloading pretrained weights, users must accept the license
        either interactively when prompted or explicitly via
        ``accept_license=True``.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        accept_license: Whether to accept the `TimesFM Non-Commercial License
            v1.0 <https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/
            LICENSE>`__ without showing the interactive license prompt.
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_multi_target: ClassVar[bool] = True
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: bool = True,
        accept_license: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)

        self.model = _TimesFM3(
            device="meta" if pretrained else device,
        )

        if pretrained:
            self._load_from_pretrained(accept_license, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        accept_license: bool,
        device: torch.device | str | None,
    ) -> TimesFM3:
        from safetensors.torch import load_file  # noqa: PLC0415

        TIMESFM_LICENSE_PROMPT = (
            "TimesFM 3.0 pretrained weights are distributed under the TimesFM "
            "Non-Commercial License v1.0 and may be used only for "
            "non-commercial, non-production purposes. Review the license at "
            "'https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/"
            "LICENSE' before downloading."
        )

        device = torch.get_default_device() if device is None else device

        path = download_checkpoint(
            repo_id="google/timesfm-3.0-pytorch",
            filename="model.safetensors",
            license_prompt=None if accept_license else TIMESFM_LICENSE_PROMPT,
        )
        ckpt = load_file(path, device=str(device))  # noqa: F841

        return self

    def forward(self, *args: Any, **kwargs: Any) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        x_context = kwargs["x_context"] if "x_context" in kwargs else args[0]
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        kwargs["_x_context_schema"] = x_context.schema

        x_query = kwargs["x_query"] if "x_query" in kwargs else args[2]
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)
        kwargs["_x_query_schema"] = x_query.schema

        x_query = expand_query(x_context.schema, x_query)

        if "x_query" in kwargs:
            kwargs["x_query"] = x_query
        else:
            args = (*args[:2], x_query, *args[3:])

        return super().forward(*args, **kwargs)  # type: ignore

    def fit(self, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        x = kwargs["x"] if "x" in kwargs else args[0]
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        kwargs["_x_context_schema"] = x.schema

        super().fit(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if self._cache is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} not yet fitted. Make sure to "
                f"call '{self.__class__.__name__}.fit()' before."
            )

        x = kwargs["x"] if "x" in kwargs else args[0]
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)

        cached_kwargs = cast(dict[str, Any], self._cache["kwargs"])
        cached_kwargs["_x_query_schema"] = x.schema
        x = expand_query(cached_kwargs["_x_context_schema"], x)

        if "x" in kwargs:
            kwargs["x"] = x
        else:
            args = (x, *args[1:])

        return super().predict(*args, **kwargs)

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, Y]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, Y * 9]

        if y_context is not None:
            columns = y_context.columns[Stype.numerical]
        else:
            assert cache is not None
            y_schema = cast(TableSchema, cache["y_schema"])
            columns = y_schema.columns[Stype.numerical]

        if x_query is not None:
            size = x_query.size()[:-1]
        else:
            assert x_context is not None
            size = (*x_context.size()[:-2], 0)

        return TableTensor(
            columns={
                Stype.numerical: [
                    f"{name}__q{i}"
                    for name, i in product(columns, range(10, 100, 10))
                ]
            },
            numerical=torch.zeros(
                (*size, len(columns) * 9),
                device=next(self.parameters()).device,
            ),
        )


class _TimesFM3(torch.nn.Module):
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
        input_transform: Name of the configured input transformation.
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
        input_transform: str = "identity",
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
                    attention_norm="rms",
                    feedforward_norm="rms",
                    qk_norm="rms",
                    use_rope_seq=True,
                    use_rope_var=False,
                    use_bias=False,
                    ff_activation="relu",
                    deterministic=True,
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
        self.input_transform = input_transform

        if use_stitching:
            self._stitching_extract_len = min(
                2 * input_patch_len,
                output_patch_len,
            )

        self.pre_transformer_resblock = ResidualBlock(
            config=residual_block_config,
            device=device,
            dtype=dtype,
        )
        self.pre_transformer_resblock.set_input_dims(
            2 * (input_patch_len + output_patch_len)
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

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable model configuration."""
        return {
            "input_patch_len": self.input_patch_len,
            "output_patch_len": self.output_patch_len,
            "quantiles": list(self.quantiles),
            "residual_block_config": asdict(self.residual_block_config),
            "transformer_config": asdict(self.transformer_config),
            "use_variate_attention": self.use_variate_attention,
            "value_clip": self.value_clip,
            "use_stitching": self.use_stitching,
            "use_linear_detrending": self.use_linear_detrending,
            "linear_detrending_threshold": (self.linear_detrending_threshold),
            "use_iterative_cpm_revin": self.use_iterative_cpm_revin,
            "use_frozen_running_stats": self.use_frozen_running_stats,
            "input_transform": self.input_transform,
        }

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
        transformer_output, _, attention_masks = self.transformer_stack(
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


def expand_query(x_context: TableSchema, x_query: TableTensor) -> TableTensor:
    """Expand the query by dummy past covariates."""
    if x_context == x_query.schema:
        return x_query

    blocks: dict[Stype, Tensor] = {}
    for stype, block in x_query.items():
        query_columns = x_query.columns[stype]
        context_columns = x_context.columns[stype]

        if query_columns == context_columns:
            blocks[stype] = block
            continue

        if not set(query_columns).issubset(context_columns):
            raise ValueError(
                "Expected query features to be a subset of context features"
            )

        if stype == Stype.numerical:
            dummy = block.new_full((*x_query.size()[:-1], 1), float("NaN"))
        else:
            raise NotImplementedError

        query_index = {column: i for i, column in enumerate(query_columns)}
        blocks[stype] = torch.cat(
            [
                dummy
                if (i := query_index.get(column)) is None
                else block.narrow(-1, i, 1)
                for column in context_columns
            ],
            dim=-1,
        )

    return TableTensor(
        columns=cast(Mapping[StypeLike, Sequence[str]], x_context.columns),
        **blocks,
    )
