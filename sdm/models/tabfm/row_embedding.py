# ruff: noqa: D101, D102

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import (
    GELU,
    Embedding,
    Linear,
    ModuleList,
    Parameter,
    RMSNorm,
    Sequential,
)

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabfm.block import TabFMTransformerBlock
from sdm.nn import InducedTransformerBlock, RotaryEmbedding


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_repeats: int,
        num_col_heads: int,
        num_row_heads: int,
        num_inducing_points: int,
        num_readout_tokens: int,
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
                Linear(1, 6, **factory_kwargs),
                GELU(approximate="tanh"),
                Linear(6, channels, **factory_kwargs),
            )

        rope = RotaryEmbedding(
            channels=channels // num_row_heads,
            layout="interleaved",
            theta=100_000,
            requires_grad=False,
            **factory_kwargs,
        )

        self.col_blocks: ModuleList[ModuleList] = ModuleList()
        self.row_blocks: ModuleList[ModuleList] = ModuleList()
        for _ in range(num_repeats):
            col_layers = ModuleList(
                InducedTransformerBlock(
                    channels=channels,
                    num_inducing_points=num_inducing_points,
                    inducing_block=TabFMTransformerBlock(
                        channels=channels,
                        num_heads=num_col_heads,
                        **factory_kwargs,
                    ),
                    output_block=TabFMTransformerBlock(
                        channels=channels,
                        num_heads=num_col_heads,
                        **factory_kwargs,
                    ),
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            )
            self.col_blocks.append(col_layers)

            row_layers = ModuleList(
                TabFMTransformerBlock(
                    channels=channels,
                    num_heads=num_row_heads,
                    rope=rope,
                    **factory_kwargs,
                )
                for _ in range(num_layers)
            )
            self.row_blocks.append(row_layers)

        self.col_projections = ModuleList(
            Sequential(
                Linear(channels, channels, **factory_kwargs),
                RMSNorm(channels, eps=1e-6, **factory_kwargs),
            )
            for _ in range(num_repeats)
        )

        self.row_norms = ModuleList(
            RMSNorm(channels, eps=1e-6, **factory_kwargs)
            for _ in range(num_repeats)
        )

        self.readout_token = Parameter(
            torch.empty((num_readout_tokens, channels), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

    def forward(
        self,
        x: Tensor,  # [..., R, C, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]

        *B, R, _, D = x.size()
        R_train = y.size(-1)
        K = self.readout_token.size(-2)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)
            else:
                assert self.y_mlp is not None
                y_emb = self.y_mlp(y.unsqueeze(-1)).unsqueeze(-2)
            x[..., :R_train, :, :] += y_emb.to(x.dtype)

        for i, (col_layers, col_proj, row_layers, row_norm) in enumerate(
            zip(
                self.col_blocks,
                self.col_projections,
                self.row_blocks,
                self.row_norms,
            )
        ):
            # Column-wise induced set attention (B * C as the batch axis).
            # Materialize once to avoid repeated copies in the column layers.
            x = x.transpose(-2, -3).contiguous()  # [..., C, R, D]
            for j, col_layer in enumerate(col_layers):
                key = f"row_embedding.col_layer{i}.{j}"
                if cache is not None and cache.is_replaying:
                    key_value = cast(KVCacheEntry, cache[key])
                else:
                    key_value = x[..., :R_train, :]

                result = col_layer(
                    query=x,  # [..., C, R, D]
                    key_value=key_value,  # [..., C, R_train, D]
                    return_key_value=cache is not None and cache.is_recording,
                )  # [..., C, R, D]

                if cache is not None and cache.is_recording:
                    x, cache[key] = result
                else:
                    x = result
                del result

            x = col_proj(x.transpose(-2, -3))  # [..., R, C, D]

            if i == 0:  # Prepend readout tokens before row-wise attention.
                x = torch.cat(
                    [
                        self.readout_token.to(x.dtype)
                        .view(*(1,) * len(B), 1, K, D)
                        .expand(*B, R, K, D),
                        x,  # [..., R, C, D]
                    ],
                    dim=-2,
                )  # [..., R, K + C, D]

            # Row-wise attention (B * R as the batch axis).
            for j, row_layer in enumerate(row_layers):
                query = x
                if i == len(self.row_blocks) - 1 and j == len(row_layers) - 1:
                    query = x[..., :K, :]

                x = row_layer(
                    query=query,  # [..., R, K + C, D] or [..., R, K, D]
                    key_value=x,  # [..., R, K + C, D]
                )  # [..., R, K + C, D] or [..., R, K, D]

            x = row_norm(x)

        return x.flatten(-2)
