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
from itertools import product
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, StypeLike, TableTensor, Task
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.timesfm3.block import ResidualBlock
from sdm.models.timesfm3.ckpt import remap_ckpt
from sdm.models.timesfm3.icl import ICLBlock
from sdm.models.timesfm3.recipe import default_recipe
from sdm.models.timesfm3.util import (
    gather_future_patches,
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
        ckpt = load_file(path, device=str(device))
        self.model.load_state_dict(remap_ckpt(ckpt, self.model), assign=True)

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
    """Assemble the TimesFM-3 patch embedding and prediction layers."""

    def __init__(
        self,
        input_patch_len: int = 32,
        output_patch_len: int = 64,
        quantiles: Sequence[float] | None = None,
        channels: int = 1280,
        num_layers: int = 20,
        num_heads: int = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.num_future_patches = output_patch_len // input_patch_len

        if quantiles is None:
            quantiles = tuple(i / 10 for i in range(1, 10))
        self.quantiles = tuple(quantiles)

        self.patch_embedding = ResidualBlock(
            in_channels=2 * (input_patch_len + output_patch_len),
            out_channels=channels,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.icl_block = ICLBlock(
            channels=channels,
            out_channels=output_patch_len * len(self.quantiles),
            num_layers=num_layers,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )

    def _prepare_patch_inputs(
        self,
        x: Tensor,  # [B, V, N, P]
        mask: Tensor,  # [B, V, N, P]
        patch_is_target: Tensor,  # [B, V, N]
        freeze_after: int | None = None,
        cpm_mask: Tensor | None = None,  # [B, N]
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Prepare masked patch features for the patch embedding.

        Running statistics use the original mask, before CPM hides target
        values. Future values use the statistics of the current patch.

        Args:
            x: Input patches with shape ``[B, V, N, P]``.
            mask: Invalid-value mask with shape ``[B, V, N, P]``.
            patch_is_target: Target indicator with shape ``[B, V, N]``.
            freeze_after: Last patch allowed to update the running mean and
                standard deviation. Counts continue accumulating.
            cpm_mask: Patch positions selected to hide target values under
                CPM, with shape ``[B, N]``.

        Returns:
            Prepared patch features with shape ``[B, V, N, 2 * (P + F)]``,
            the fully masked patch indicator with shape ``[B, V, N]``, and
            ``(count, mean, std)``. Here ``F = num_future_patches * P``.
        """
        count, mean, std = get_running_stats(x, mask)

        if freeze_after is not None and 0 <= freeze_after < x.size(2) - 1:
            mean[:, :, freeze_after + 1 :] = mean[
                :, :, freeze_after : freeze_after + 1
            ]
            std[:, :, freeze_after + 1 :] = std[
                :, :, freeze_after : freeze_after + 1
            ]

        current_mask = mask
        if cpm_mask is not None:
            current_mask = current_mask | (
                cpm_mask[:, None, :, None] & patch_is_target[..., None]
            )

        future_x, past_end_mask = gather_future_patches(
            x, self.num_future_patches
        )
        future_mask, _ = gather_future_patches(
            current_mask, self.num_future_patches
        )
        future_mask |= patch_is_target[..., None]
        future_mask |= past_end_mask

        patch_mask = current_mask.all(dim=-1) & future_mask.all(dim=-1)

        P, F = x.size(-1), future_x.size(-1)
        patch_features = x.new_empty(
            (*x.shape[:-1], 2 * (P + F)),  # (B, V, N, 2 * (P + F))
            dtype=self.patch_embedding.res.weight.dtype,
        )

        (
            current_values,
            future_values,
            current_mask_values,
            future_mask_values,
        ) = patch_features.split(
            (P, F, P, F),
            dim=-1,
        )
        current_values.copy_(
            revin(x, mean, std).masked_fill_(current_mask, 0.0)
        )
        future_values.copy_(
            revin(future_x, mean, std).masked_fill_(future_mask, 0.0)
        )
        current_mask_values.copy_(current_mask)
        future_mask_values.copy_(future_mask)

        return patch_features, patch_mask, (count, mean, std)

    def _preprocess(
        self,
        x: Tensor,
        mask: Tensor,
        patch_is_target: Tensor,
        freeze_after: int | None = None,
        cpm_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Embed prepared patches and retain statistics for decoding."""
        patch_features, patch_mask, stats = self._prepare_patch_inputs(
            x, mask, patch_is_target, freeze_after, cpm_mask
        )
        embeddings = self.patch_embedding(patch_features)

        return embeddings, patch_features, patch_mask, stats


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
