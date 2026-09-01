# ruff: noqa: D101, D102

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, Linear, ModuleList, RMSNorm, Sequential

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
        num_key_value_heads_test: int | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_key_value_heads_test = num_key_value_heads_test

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, bias=False, **factory_kwargs)

        self.layers = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                query_log_scale=True,
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
        R_test = x.size(-2) - R_train
        kv_heads = self.num_key_value_heads_test

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))  # [..., R_train, D]

            x[..., :R_train, :] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            key = f"icl_block.layer{i}"
            last_layer = i == len(self.layers) - 1
            query = x[..., R_train:, :] if last_layer else x
            is_recording = cache is not None and cache.is_recording

            if kv_heads is None or R_test == 0:
                result = layer(
                    query=query,
                    key_value=(
                        cast(KVCacheEntry, cache[key])
                        if cache is not None and cache.is_replaying
                        else x[..., :R_train, :]
                    ),
                    return_key_value=is_recording,
                    batch_size_limit=batch_size_limit,
                )

                if is_recording:
                    x, key_value = result
                    if kv_heads is not None:
                        key_value = KVCacheEntry(
                            key=key_value.key[..., :kv_heads, :].contiguous(),
                            value=key_value.value[
                                ..., :kv_heads, :
                            ].contiguous(),
                        )
                    cache[key] = key_value
                else:
                    x = result
                continue

            if cache is not None and cache.is_replaying:
                x = layer(
                    query=query,
                    key_value=cast(KVCacheEntry, cache[key]),
                    batch_size_limit=batch_size_limit,
                )
                continue

            train = x[..., :R_train, :]
            test = x[..., R_train:, :]
            train_out, key_value = layer(
                query=x[..., :0, :] if last_layer else train,
                key_value=train,
                return_key_value=True,
                batch_size_limit=batch_size_limit,
            )
            key_value = KVCacheEntry(
                key=key_value.key[..., :kv_heads, :].contiguous(),
                value=key_value.value[..., :kv_heads, :].contiguous(),
            )
            test_out = layer(
                query=test,
                key_value=key_value,
                batch_size_limit=batch_size_limit,
            )
            x = (
                test_out
                if last_layer
                else torch.cat((train_out, test_out), -2)
            )

            if is_recording:
                cache[key] = key_value

        return self.head(self.norm(x))
