# ruff: noqa: D101, D102

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, Linear, ModuleList, RMSNorm, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock


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

        self.num_classes = num_classes
        self.y_lin = Linear(
            num_classes or 1,
            channels,
            bias=num_classes > 0,
            **factory_kwargs,
        )
        self.layers = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(channels, **factory_kwargs)
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, out_channels, **factory_kwargs),
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, out_channels]
        R_train = y.size(-1)
        if y.numel() > 0:
            if self.num_classes > 0:
                y = F.one_hot(
                    y.long(),
                    num_classes=self.num_classes,
                )
            else:
                y = y.unsqueeze(-1)
            y_emb = self.y_lin(y.to(self.y_lin.weight.dtype)).to(x.dtype)
            x = torch.cat(
                [x[..., :R_train, :] + y_emb, x[..., R_train:, :]],
                dim=-2,
            )

        for i, layer in enumerate(self.layers):
            key = f"icl_block.layer{i}"
            result = layer(
                query=(
                    x[..., R_train:, :] if i == len(self.layers) - 1 else x
                ),
                key_value=(
                    cast(KVCacheEntry, cache[key])
                    if cache is not None and cache.is_replaying
                    else x[..., :R_train, :]
                ),
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit=batch_size_limit,
            )
            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        return self.head(self.norm(x))
