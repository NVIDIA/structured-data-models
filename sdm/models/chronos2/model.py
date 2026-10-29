# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.chronos2.block import ResidualBlock
from sdm.models.chronos2.ckpt import remap_ckpt
from sdm.models.chronos2.icl import ICLBlock
from sdm.models.chronos2.recipe import default_recipe
from sdm.models.chronos2.util import rollout_quantiles
from sdm.nn import RevIN

QUANTILES = (0.01, *[i / 100 for i in range(5, 100, 5)], 0.99)


class Chronos2(ICLModel):
    """The time series forecasting model from the
    `"Chronos-2: From Univariate to Universal Forecasting"
    <https://arxiv.org/abs/2510.15821>`_ paper.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
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
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        return super().forward(
            x_context[..., -self.model.context_length :, :],
            y_context[..., -self.model.context_length :, :],
            *args,
            **kwargs,
        )

    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r""":meta private:"""  # noqa: D415
        super().fit(
            x[..., -self.model.context_length :, :],
            y[..., -self.model.context_length :, :],
            *args,
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
        self.model.load_state_dict(state, strict=True, assign=True)

        return self

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:

        if cache is not None and not cache.is_recording:
            x_context = cast(TableTensor, cache["x_context"])
            y_context = cast(TableTensor, cache["y_context"])
            x_context_only = cast(TableTensor | None, cache["x_context_only"])
        assert x_context is not None
        assert y_context is not None
        if cache is not None and cache.is_recording:
            cache["x_context"] = x_context
            cache["y_context"] = y_context
            cache["x_context_only"] = x_context_only
        y = y_context.numerical
        columns = [
            f"{name}__q{round(q * 100)}"
            for name in y_context.columns[Stype.numerical]
            for q in QUANTILES
        ]
        if x_query is None or x_query.size(-2) == 0:
            out = y.new_empty((*y.size()[:-2], 0, len(columns)))
        else:
            past = (
                y.new_empty((*y.size()[:-1], 0))
                if x_context_only is None
                else x_context_only.numerical
            )
            context = torch.cat([y, past, x_context.numerical], dim=-1)
            *batch, length, variates = context.shape
            context = context.transpose(-2, -1).reshape(-1, length).float()
            horizon = x_query.size(-2)
            future = context.new_full((*batch, horizon, variates), torch.nan)
            known = x_query.numerical
            future[..., y.size(-1) + past.size(-1) :] = known
            future = future.transpose(-2, -1).reshape(-1, horizon)
            groups = torch.arange(
                context.size(0) // variates, device=context.device
            ).repeat_interleave(variates)
            out = self.model(context, future, groups, horizon)
            out = out.reshape(*batch, variates, len(QUANTILES), horizon)
            out = out[..., : y.size(-1), :, :].movedim(-1, -3).flatten(-2)
        return TableTensor.from_tensor(out, columns=columns)


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
        self.norm = RevIN(eps=0)
        self.patch_embedding = ResidualBlock(
            in_channels=3 * patch_size,
            hidden_channels=hidden_channels,
            out_channels=channels,
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
        self.head = ResidualBlock(
            in_channels=channels,
            hidden_channels=hidden_channels,
            out_channels=len(QUANTILES) * patch_size,
            **factory_kwargs,
        )

    def forward(
        self,
        context: Tensor,  # [S, T]
        future: Tensor,  # [S, H], NaNs mark unknown future values.
        group_ids: Tensor,  # [S]
        prediction_length: int,
    ) -> Tensor:  # [S, Q, H]
        outputs = []
        remaining = prediction_length
        patches = min(
            (remaining + self.patch_size - 1) // self.patch_size,
            self.max_output_patches,
        )
        prediction = self._predict(context, future, group_ids, patches)
        outputs.append(prediction)
        remaining -= prediction.size(-1)
        if remaining > 0:
            paths = 9
            # Rollout feedback and refitted statistics use the linear scale.
            prediction = prediction.sinh()
            context, future = context.sinh(), future.sinh()
            context = context[:, None, :].expand(-1, paths, -1)
            future = future[:, None, :].expand(-1, paths, -1)
            group_ids = group_ids[:, None] * paths + torch.arange(
                paths, device=group_ids.device
            )
        while remaining > 0:
            # Default rollout levels 0.1, ..., 0.9 are every second knot.
            feedback = prediction[:, 2:19:2]
            context = torch.cat([context, feedback], dim=-1)
            context = context[..., -self.context_length :]
            known = future[..., : prediction.size(-1)]
            context[..., -known.size(-1) :] = torch.where(
                known.isnan(), context[..., -known.size(-1) :], known
            )
            future = future[..., prediction.size(-1) :]
            patches = min(
                (remaining + self.patch_size - 1) // self.patch_size,
                self.max_output_patches,
            )
            # Refit only after extending the recipe-normalized context.
            x = context.flatten(0, 1)
            mean = x.nanmean(-1, keepdim=True).nan_to_num(nan=0.0)
            scale = (x - mean).square().nanmean(-1, keepdim=True).sqrt()
            scale = scale.nan_to_num(nan=1.0)
            scale = scale.where(scale != 0, 1e-5)
            samples = self._predict(
                self.norm(x, mean, scale).asinh(),
                self.norm(future.flatten(0, 1), mean, scale).asinh(),
                group_ids.flatten(),
                patches,
            )
            samples = self.norm.inverse(
                samples.sinh(), mean.unsqueeze(-1), scale.unsqueeze(-1)
            )
            samples = samples.unflatten(0, (-1, 9)).flatten(1, 2)
            prediction = rollout_quantiles(samples, QUANTILES)
            outputs.append(prediction.asinh())
            remaining -= prediction.size(-1)
        return torch.cat(outputs, dim=-1)[..., :prediction_length]

    def _predict(
        self,
        context: Tensor,
        future: Tensor,
        group_ids: Tensor,
        num_output_patches: int,
    ) -> Tensor:
        context = context[..., -self.context_length :].float()
        x = context
        observed = ~context.isnan()
        padding = -context.size(-1) % self.patch_size
        x = F.pad(x, (padding, 0)).unflatten(-1, (-1, self.patch_size))
        observed = F.pad(observed, (padding, 0))
        observed = observed.unflatten(-1, (-1, self.patch_size))
        x = x.masked_fill(~observed, 0)
        length = num_output_patches * self.patch_size
        future = future[..., :length].float()
        future = F.pad(future, (0, length - future.size(-1)), value=torch.nan)
        future_mask = ~future.isnan()
        future = future.masked_fill(~future_mask, 0)
        future = future.unflatten(-1, (num_output_patches, self.patch_size))
        future_mask = future_mask.unflatten(
            -1, (num_output_patches, self.patch_size)
        )
        context_length = x.size(-2) * self.patch_size
        time = (
            torch.arange(
                -context_length, length, device=x.device, dtype=torch.float32
            )
            / 8192
        )
        time = time.unflatten(-1, (-1, self.patch_size))
        values = torch.cat([x, future], dim=-2).to(self.reg_token.dtype)
        mask = torch.cat([observed, future_mask], dim=-2)
        time = time.to(values.dtype).expand_as(values)
        x = self.patch_embedding(
            torch.cat([time, values, mask.to(values.dtype)], dim=-1)
        )
        first = observed.size(-2)
        x = torch.cat(
            [
                x[:, :first],
                self.reg_token.expand(x.size(0), 1, -1),
                x[:, first:],
            ],
            dim=-2,
        )
        attention_mask = torch.cat(
            [
                observed.any(-1),
                observed.new_ones((x.size(0), 1 + num_output_patches)),
            ],
            dim=-1,
        )
        x = self.icl_block(x, attention_mask, group_ids)
        out = self.head(x[:, -num_output_patches:])
        out = out.unflatten(-1, (len(QUANTILES), self.patch_size))
        return out.transpose(-3, -2).flatten(-2).float()
