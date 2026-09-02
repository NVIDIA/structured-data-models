# ruff: noqa: D101, D102

import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Parameter

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock, _RMSNorm
from sdm.nn import InducedTransformerBlock, RotaryEmbedding

_COL_BATCH_SIZE_LIMIT = 50
_ROW_BATCH_SIZE_LIMIT = 10_000


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, bias=False, **factory_kwargs)

        self.readout_token = Parameter(
            torch.empty(num_readout_tokens, channels, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=100_000,
            requires_grad=False,
            partial_rotary_factor=0.25,
            **factory_kwargs,
        )

        self.col_blocks = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    query_log_scale=True,
                    **factory_kwargs,
                ),
                output_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    query_log_scale=False,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.row_blocks = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                query_log_scale=False,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = _RMSNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, K + C, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]

        *B, R, _, D = x.size()
        batch_size = max(1, math.prod(B))
        R_train = y.size(-1)
        K = self.readout_token.size(-2)

        readout_token = self.readout_token.to(x.dtype)
        readout_token = readout_token.view(*(1,) * len(B), 1, K, D)
        x[..., :, :K, :].copy_(readout_token.expand(*B, R, K, D))

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)
            x[..., :R_train, K:, :] += y_emb.to(x.dtype)

        for i, (col_block, row_block) in enumerate(
            zip(self.col_blocks, self.row_blocks)
        ):
            key = f"row_embedding.col_block{i}"
            features = x[..., :, K:, :]
            num_cols = features.size(-2)
            cols_per_chunk = max(1, _COL_BATCH_SIZE_LIMIT // batch_size)

            key_out: Tensor | None = None
            value_out: Tensor | None = None
            cached_key_value: KVCacheEntry | None = None
            if cache is not None and cache.is_replaying:
                cached_key_value = cast(KVCacheEntry, cache[key])

            for start in range(0, num_cols, cols_per_chunk):
                end = min(start + cols_per_chunk, num_cols)
                query = features[..., :, start:end, :]
                query = query.transpose(-2, -3)

                if cached_key_value is None:
                    key_value: Tensor | KVCacheEntry = query[..., :R_train, :]
                else:
                    key_value = KVCacheEntry(
                        key=cached_key_value.key[..., start:end, :, :, :],
                        value=cached_key_value.value[..., start:end, :, :, :],
                    )

                result = col_block(
                    query=query,
                    key_value=key_value,
                    return_key_value=cache is not None and cache.is_recording,
                    batch_size_limit=_COL_BATCH_SIZE_LIMIT,
                )

                if cache is not None and cache.is_recording:
                    query, chunk_key_value = result
                    if key_out is None:
                        key_size = chunk_key_value.key.size()
                        value_size = chunk_key_value.value.size()
                        key_out = chunk_key_value.key.new_empty(
                            *key_size[:-4],
                            num_cols,
                            *key_size[-3:],
                        )
                        value_out = chunk_key_value.value.new_empty(
                            *value_size[:-4],
                            num_cols,
                            *value_size[-3:],
                        )
                    assert value_out is not None
                    key_out[..., start:end, :, :, :] = chunk_key_value.key
                    value_out[..., start:end, :, :, :] = chunk_key_value.value
                    del chunk_key_value
                else:
                    query = result
                assert isinstance(query, Tensor)

                features[..., :, start:end, :].copy_(query.transpose(-2, -3))
                del query, key_value, result

            if cache is not None and cache.is_recording:
                assert key_out is not None
                assert value_out is not None
                cache[key] = KVCacheEntry(key=key_out, value=value_out)

            rows_per_chunk = max(1, _ROW_BATCH_SIZE_LIMIT // batch_size)
            last_layer = i == len(self.row_blocks) - 1
            for start in range(0, R, rows_per_chunk):
                end = min(start + rows_per_chunk, R)
                key_value = x[..., start:end, :, :]
                query = key_value[..., :K, :] if last_layer else key_value
                result = row_block(
                    query=query,
                    key_value=key_value,
                    batch_size_limit=_ROW_BATCH_SIZE_LIMIT,
                )

                if last_layer:
                    x[..., start:end, :K, :].copy_(result)
                else:
                    key_value.copy_(result)
                del query, key_value, result

            if last_layer:
                x = x[..., :, :K, :].contiguous()

        return self.norm(x).flatten(-2)
