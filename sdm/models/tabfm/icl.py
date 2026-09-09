# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, Linear, ModuleList, RMSNorm, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabfm.block import TabFMTransformerBlock


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.y_emb: torch.nn.Module | None = None
        self.y_mlp: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_mlp = Sequential(
                Linear(1, 2 * channels, **factory_kwargs),
                GELU(approximate="tanh"),
                Linear(2 * channels, channels, **factory_kwargs),
            )

        self.layers = ModuleList(
            TabFMTransformerBlock(channels, num_heads, **factory_kwargs)
            for _ in range(num_layers)
        )

        self.norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(approximate="tanh"),
            Linear(2 * channels, out_channels, **factory_kwargs),
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, out_channels]
        R_train = y.size(-1)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_mlp is not None
                y_emb = self.y_mlp(y.unsqueeze(-1))  # [..., R_train, D]
            x[..., :R_train, :] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            key = f"icl_block.layer{i}"
            result = layer(
                query=x[..., R_train:, :] if i == len(self.layers) - 1 else x,
                key_value=(
                    cast(KVCacheEntry, cache[key])
                    if cache is not None and cache.is_replaying
                    else x[..., :R_train, :]
                ),
                return_key_value=cache is not None and cache.is_recording,
                out=None
                if torch.is_grad_enabled()
                else x[..., R_train:, :]
                if i == len(self.layers) - 1
                else x,
            )  # [..., R, D] or [..., R_test, D]

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result
            del result

        return self.head(self.norm(x))  # [..., R_test, out_channels]
