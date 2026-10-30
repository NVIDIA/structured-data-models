# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import math
from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import ReLU

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.chronos2.ckpt import remap_ckpt
from sdm.models.chronos2.icl import ICLBlock
from sdm.models.chronos2.recipe import default_recipe
from sdm.models.chronos2.util import rollout_quantiles
from sdm.nn import ResidualMLP
from sdm.tensor.table import TableSchema

QUANTILES = (0.01, *[i / 100 for i in range(5, 100, 5)], 0.99)


class Chronos2(ICLModel):
    """The time series forecasting model from the
    `"Chronos-2: From Univariate to Universal Forecasting"
    <https://arxiv.org/abs/2510.15821>`_ paper.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.

    Additional Forward Parameters:
        cross_learning (bool): Whether examples in a batch will share
            information through variate attention. (default: ``False``)
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_multi_target: ClassVar[bool] = True
    supports_context_only_columns: ClassVar[bool] = True
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)

        self.model = _Chronos2(device="meta" if pretrained else device)

        if pretrained:
            self._load_from_pretrained(device)

        self.eval()

    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        *args: Any,
        cross_learning: bool = False,
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if cross_learning:  # Keep estimator groups separate:
            kwargs["estimator_batch_size"] = 1
        return super().forward(
            x_context[..., -self.model.context_length :, :],
            y_context[..., -self.model.context_length :, :],
            *args,
            cross_learning=cross_learning,
            **kwargs,
        )

    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        *args: Any,
        cross_learning: bool = False,
        **kwargs: Any,
    ) -> None:
        r""":meta private:"""  # noqa: D415
        if cross_learning:  # Keep estimator groups separate:
            kwargs["estimator_batch_size"] = 1
        super().fit(
            x[..., -self.model.context_length :, :],
            y[..., -self.model.context_length :, :],
            *args,
            cross_learning=cross_learning,
            **kwargs,
        )

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        device: torch.device | str | None,
    ) -> Chronos2:
        device = torch.get_default_device() if device is None else device

        path = download_checkpoint(
            repo_id="amazon/chronos-2",
            filename="model.safetensors",
            revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        )
        state = remap_ckpt(load_file(path, device=str(device)))
        self.model.load_state_dict(state, assign=True)

        return self

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, Y]
        x_query: TableTensor | None,  # [..., R_query, D]
        x_context_only: TableTensor | None,  # [..., R_context, D_only]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        cross_learning: bool = False,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, Y * 21]

        if cache is None or cache.is_recording:
            assert x_context is not None
            x_context: Tensor = x_context.numerical
            assert y_context is not None
            y = y_context.numerical
            if x_context_only is not None:
                x_context_only: Tensor = x_context_only.numerical
            else:
                x_context_only: Tensor = y.new_empty((*y.size()[:-1], 0))
            y_schema = y_context.schema
        else:
            x_context: Tensor = cast(Tensor, cache["x_context"])
            y = cast(Tensor, cache["y"])
            x_context_only: Tensor = cast(Tensor, cache["x_context_only"])
            y_schema = cast(TableSchema, cache["y_schema"])

        if cache is not None and cache.is_recording:
            cache["x_context"] = x_context
            cache["x_context_only"] = x_context_only
            cache["y"] = y
            cache["y_schema"] = y_schema

        if x_query is None or x_query.size(-2) == 0:
            out = y.new_empty((*y.size()[:-2], 0, y.size(-1) * len(QUANTILES)))
        else:
            context = torch.cat([y, x_context_only, x_context], dim=-1)
            context = context.transpose(-2, -1)
            query = context.new_full(
                (*context.shape[:-1], x_query.size(-2)), torch.nan
            )
            query[..., y.size(-1) + x_context_only.size(-1) :, :] = (
                x_query.numerical.transpose(-2, -1)
            )
            shape = context.size()[:-1]
            if cross_learning:
                context = context.flatten(0, -2)
                query = query.flatten(0, -2)
            out = self.model(context, query)
            if cross_learning:
                out = out.unflatten(-3, shape)
            out = out[..., : y.size(-1), :, :].movedim(-1, -3).flatten(-2)

        return TableTensor(
            columns={
                Stype.numerical: [
                    f"{name}__q{round(q * 100)}"
                    for name in y_schema.columns[Stype.numerical]
                    for q in QUANTILES
                ],
            },
            numerical=out,
        )


class _Chronos2(torch.nn.Module):
    def __init__(
        self,
        channels: int = 768,
        hidden_channels: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        context_length: int = 8192,
        max_output_patches: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.patch_size = patch_size
        self.context_length = context_length
        self.max_output_patches = max_output_patches

        self.patch_embedding = ResidualMLP(
            in_channels=3 * patch_size,
            hidden_channels=hidden_channels,
            out_channels=channels,
            activation=ReLU(),
            **factory_kwargs,
        )
        self.reg_token = torch.nn.Parameter(
            torch.zeros(1, channels, **factory_kwargs)
        )
        self.icl_block = ICLBlock(
            channels=channels,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            **factory_kwargs,
        )
        self.head = ResidualMLP(
            in_channels=channels,
            hidden_channels=hidden_channels,
            out_channels=len(QUANTILES) * patch_size,
            activation=ReLU(),
            **factory_kwargs,
        )

    def forward(
        self,
        context: Tensor,  # [..., C, R_context]
        query: Tensor,  # [..., C, R_query]
    ) -> Tensor:  # [..., C, Q, R_query]

        prediction_size = remaining = query.size(-1)
        num_output_patches = min(
            math.ceil(remaining / self.patch_size),
            self.max_output_patches,
        )
        out = self._predict(context, query, num_output_patches)
        remaining -= out.size(-1)
        outs = [out]

        if remaining > 0:
            # Revert to original space:
            context, query, out = context.sinh(), query.sinh(), out.sinh()
            num_paths = 9
            context = context.unsqueeze(-3).expand(
                *context.size()[:-2], num_paths, *context.size()[-2:]
            )
            query = query.unsqueeze(-3).expand(
                *query.size()[:-2], num_paths, *query.size()[-2:]
            )

        while remaining > 0:
            # Default rollout levels 0.1, ..., 0.9 are every second knot.
            out = out[..., 2:19:2, :].transpose(-3, -2)
            known = query[..., : out.size(-1)]
            out = out.where(known.isnan(), known)
            context = torch.cat([context, out], dim=-1)
            context = context[..., -self.context_length :]
            query = query[..., out.size(-1) :]

            num_output_patches = min(
                math.ceil(remaining / self.patch_size),
                self.max_output_patches,
            )
            mean = context.nanmean(dim=-1, keepdim=True).nan_to_num(0.0)
            scale = (context - mean).square().nanmean(dim=-1, keepdim=True)
            scale = scale.sqrt().nan_to_num(1.0)
            scale = scale.where(scale != 0, 1e-5)
            out = self._predict(
                (context - mean).div_(scale).asinh_(),
                (query - mean).div_(scale).asinh_(),
                num_output_patches,
            )
            out.sinh_().mul_(scale.unsqueeze(-1)).add_(mean.unsqueeze(-1))
            out = out.transpose(-4, -3).flatten(-3, -2)
            out = rollout_quantiles(out, QUANTILES)
            outs.append(out.asinh())
            remaining -= out.size(-1)

        return torch.cat(outs, dim=-1)[..., :prediction_size]

    def _predict(
        self,
        context: Tensor,  # [..., C, R_context]
        query: Tensor,  # [..., C, R_query]
        num_output_patches: int,
    ) -> Tensor:  # [..., C, Q, num_output_patches * patch_size]

        context = context[..., -self.context_length :]
        context_mask = ~context.isnan()
        context = context.masked_fill(~context_mask, 0)
        padding = -context.size(-1) % self.patch_size
        context = F.pad(context, (padding, 0))
        context = context.unflatten(-1, (-1, self.patch_size))
        context_mask = F.pad(context_mask, (padding, 0))
        context_mask = context_mask.unflatten(-1, (-1, self.patch_size))
        context_size = context.size(-2) * self.patch_size

        query_size = num_output_patches * self.patch_size
        query = query[..., :query_size]
        query = F.pad(query, (0, query_size - query.size(-1)), value=torch.nan)
        query_mask = ~query.isnan()
        query = query.masked_fill(~query_mask, 0)
        query = query.unflatten(-1, (num_output_patches, self.patch_size))
        query_mask = query_mask.unflatten(
            -1, (num_output_patches, self.patch_size)
        )

        time = torch.arange(
            -context_size,
            query_size,
            device=context.device,
            dtype=context.dtype,
        )
        time = (time / 8192).view(-1, self.patch_size)

        x = torch.cat([context, query], dim=-2)
        mask = torch.cat([context_mask, query_mask], dim=-2)
        time = time.expand_as(x)

        x = torch.cat([time, x, mask.to(x.dtype)], dim=-1)
        x = self.patch_embedding(x)

        x = torch.cat(  # Inject [REG] token:
            [
                x[..., : context.size(-2), :],
                self.reg_token.expand(*x.size()[:-2], 1, -1),
                x[..., context.size(-2) :, :],
            ],
            dim=-2,
        )

        patch_mask = ~context_mask.any(dim=-1)
        if not patch_mask.any():  # Pay for device sync here:
            patch_mask = None
        else:
            query_patch_mask = patch_mask.new_zeros(
                *x.size()[:-2], 1 + num_output_patches
            )
            patch_mask = torch.cat([patch_mask, query_patch_mask], dim=-1)

        x = self.icl_block(x, patch_mask)
        out = self.head(x[..., -num_output_patches:, :])
        out = out.unflatten(-1, (len(QUANTILES), self.patch_size))
        return out.transpose(-3, -2).flatten(-2).float()
