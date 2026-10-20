# Copyright 2026 The MatrixFormer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
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
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, Linear, ModuleDict, Parameter, Sequential

from sdm import TableTensor
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import CompletionModel
from sdm.models.matrixformer.block import AxialBlock
from sdm.models.matrixformer.ckpt import remap_ckpt

_EXPERTS: dict[str, tuple[str, ...]] = {
    "plus": ("impute", "recsys"),
    "impute": ("impute",),
    "recsys": ("recsys",),
}


class MatrixFormer(CompletionModel):
    r"""The matrix completion foundation model from the `"MatrixFormer: A
    Foundation Model for Matrix Completion" <https://arxiv.org/abs/2610.06751>`__
    paper.

    :class:`MatrixFormer` embeds every matrix entry as a token, surrounds the
    grid with learned row and column CLS tokens, and alternates attention
    across columns within each row and across rows within each column.
    A decoder predicts a distribution over 5000 value bins per entry, whose
    mean completes the missing entries.

    MatrixFormer+ (``variant="plus"``) runs a general expert and a
    recommender-system expert, and blends their predictions with one convex
    weight per matrix fitted to minimize the reconstruction error on the
    observed entries.

    Args:
        variant: The pretrained model to use.
            ``"plus"`` blends both experts, ``"impute"`` runs the general
            expert, and ``"recsys"`` runs the recommender-system expert.
        pretrained: Whether to load the pretrained checkpoints.
        device: The device.
    """

    def __init__(
        self,
        variant: Literal["plus", "impute", "recsys"] = "plus",
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.variant = variant
        self.experts = ModuleDict(
            {
                name: _MatrixFormer(device="meta" if pretrained else device)
                for name in _EXPERTS[variant]
            }
        )

        if pretrained:
            self._load_from_pretrained(device=device)

        self.eval()

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        args = [f"variant={self.variant!r}"]
        if device.type != "cpu":
            args.append(f"device={device}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    def _load_from_pretrained(
        self,
        device: torch.device | str | None,
    ) -> MatrixFormer:
        from safetensors.torch import load_file  # noqa: PLC0415

        device = torch.get_default_device() if device is None else device

        for name, expert in self.experts.items():
            path = download_checkpoint(
                "Tabimpute/TabImpute",
                f"matrixformer/{name}/model.safetensors",
            )
            ckpt = remap_ckpt(load_file(path, device=str(device)))
            expert.load_state_dict(ckpt, assign=True)

        return self

    def _forward(
        self,
        x: TableTensor,  # [..., R, C]
        *,
        max_rows: int | None = 2000,
        max_cols: int | None = 1000,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R, C]
        r"""Predict every entry of a standardized table.

        Args:
            x: The standardized table, whose missing entries are ``NaN``.
            max_rows: The maximum number of rows attended jointly.
                Larger tables are processed in overlapping row windows.
                ``None`` attends over all rows.
            max_cols: The maximum number of columns attended jointly.
                ``None`` attends over all columns.
            kwargs: Unused keyword arguments.
        """
        values = x.numerical
        preds = [
            _windowed(expert, values, max_rows, max_cols)
            for expert in self.experts.values()
        ]

        if len(preds) == 1:
            return x.replace_blocks(numerical=preds[0].to(values.dtype))

        # Fit one convex weight per matrix on the observed entries. Columns are
        # standardized, so squared errors are already column-normalized.
        general, specialist = preds
        observed = values.isfinite()
        target = values.nan_to_num()
        general_err = (general - target).where(observed, 0.0)
        diff = (specialist - target).where(observed, 0.0) - general_err
        num = -(general_err * diff).sum(dim=(-2, -1))  # [...]
        den = diff.square().sum(dim=(-2, -1))  # [...]
        weight = (num / den).where(den > 0, 0.0).clamp(0.0, 1.0)
        weight = weight[..., None, None]  # [..., 1, 1]

        out = general.lerp(specialist, weight)
        return x.replace_blocks(numerical=out.to(values.dtype))


class _MatrixFormer(torch.nn.Module):
    borders: Tensor

    def __init__(
        self,
        channels: int = 768,
        hidden_channels: int = 1536,
        num_layers: int = 8,
        num_heads: int = 16,
        num_cls: int = 12,
        num_bins: int = 5000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_cls = num_cls

        self.value_mlp = Sequential(
            Linear(1, channels, **factory_kwargs),
            GELU(),
            Linear(channels, channels, **factory_kwargs),
        )
        self.mask_token = Parameter(torch.zeros(channels, **factory_kwargs))
        self.missing_embedding = Embedding(2, channels, **factory_kwargs)
        self.row_cls = Embedding(num_cls, channels, **factory_kwargs)
        self.col_cls = Embedding(num_cls, channels, **factory_kwargs)

        self.blocks = torch.nn.ModuleList(
            AxialBlock(
                channels=channels,
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.decoder = Sequential(
            Linear(channels, hidden_channels, **factory_kwargs),
            GELU(),
            Linear(hidden_channels, hidden_channels, **factory_kwargs),
            GELU(),
            Linear(hidden_channels, num_bins, **factory_kwargs),
        )

        # Bin borders of the predictive distribution in standardized units.
        self.register_buffer(
            "borders",
            torch.linspace(-10.0, 10.0, num_bins + 1, device=device),
        )

    def forward(self, x: Tensor) -> Tensor:  # [..., R, C] -> [..., R, C]
        k = self.num_cls
        missing = x.isnan()  # [..., R, C]

        h = self.value_mlp(x.masked_fill(missing, 0.0).unsqueeze(-1))
        h = h.where(~missing.unsqueeze(-1), self.mask_token)  # [..., R, C, F]
        h = h + self.missing_embedding(missing.long())

        # Surround the grid with `k` rows of column CLS tokens and `k` columns
        # of row CLS tokens; corner tokens sum both.
        batch, (num_rows, num_cols) = h.size()[:-3], h.size()[-3:-1]
        row_cls = self.row_cls.weight.to(h.dtype)  # [K, F]
        col_cls = self.col_cls.weight.to(h.dtype)  # [K, F]
        corner = col_cls[:, None] + row_cls[None, :]  # [K, K, F]
        top = torch.cat(
            [
                corner.expand(*batch, k, k, -1),
                col_cls[:, None].expand(*batch, k, num_cols, -1),
            ],
            dim=-2,
        )
        left = row_cls[None].expand(*batch, num_rows, k, -1)
        h = torch.cat([top, torch.cat([left, h], dim=-2)], dim=-3)

        # CLS tokens sit at position zero, which RoPE leaves unrotated.
        row_pos = _positions(num_rows, k, x.device)
        col_pos = _positions(num_cols, k, x.device)
        for block in self.blocks:
            h = block(h, row_pos=row_pos, col_pos=col_pos)

        logits = self.decoder(h[..., k:, k:, :])  # [..., R, C, O]
        with torch.amp.autocast(x.device.type, enabled=False):
            probs = logits.float().softmax(dim=-1)
            return probs @ _bin_means(self.borders.float())  # [..., R, C]


def _positions(size: int, num_cls: int, device: torch.device) -> Tensor:
    pos = torch.arange(-num_cls, size, device=device, dtype=torch.float32)
    return pos.clamp_min_(0.0)  # [K + size]


def _bin_means(borders: Tensor) -> Tensor:  # [O + 1] -> [O]
    # Interior bins use their midpoint; the outer bins are half-normal tails
    # whose median equals the bin width.
    widths = borders.diff()
    means = borders[:-1] + 0.5 * widths
    tail = math.sqrt(2.0 / math.pi) / 0.6744897501960817
    means[0] = borders[1] - widths[0] * tail
    means[-1] = borders[-2] + widths[-1] * tail
    return means


def _spans(size: int, limit: int | None) -> list[tuple[int, int]]:
    # Consecutive windows of `limit` entries; the last window ends at `size`.
    if limit is None or limit >= size:
        return [(0, size)]
    spans = [(start, start + limit) for start in range(0, size - limit, limit)]
    return [*spans, (size - limit, size)]


def _windowed(
    model: torch.nn.Module,
    x: Tensor,  # [..., R, C]
    max_rows: int | None,
    max_cols: int | None,
) -> Tensor:  # [..., R, C]
    row_spans = _spans(x.size(-2), max_rows)
    col_spans = _spans(x.size(-1), max_cols)
    if len(row_spans) == len(col_spans) == 1:
        return model(x)

    total = x.new_zeros(x.size(), dtype=torch.float32)
    count = x.new_zeros(x.size()[-2:], dtype=torch.float32)
    for row_start, row_end in row_spans:
        for col_start, col_end in col_spans:
            window = (
                ...,
                slice(row_start, row_end),
                slice(col_start, col_end),
            )
            total[window] += model(x[window])
            count[window[1:]] += 1.0
    return total / count
