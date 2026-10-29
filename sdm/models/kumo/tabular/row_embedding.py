# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Parameter

from sdm._token_buffer import TokenBuffer, add_labels, run_column_layers
from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.cell_embedding import CellEmbedding
from sdm.nn import (
    GatedLogScale,
    InducedTransformerBlock,
    LogScale,
    RMSNorm,
    RotaryEmbedding,
)


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        group_size: int,
        num_frequencies: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.channels = channels

        self.cell_embedding = CellEmbedding(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )

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
            **factory_kwargs,
        )

        self.col_blocks = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    query_scaling=LogScale(
                        num_heads=num_heads,
                        **factory_kwargs,
                    ),
                    **factory_kwargs,
                ),
                output_block=KumoTabularTransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    query_scaling=None,
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
                query_scaling=GatedLogScale(
                    channels=channels // num_heads,
                    num_heads=num_heads,
                    hidden_channels=64,
                    **factory_kwargs,
                ),
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
        num_columns: Tensor | None = None,
        offload: Literal["cpu"] | None = None,
    ) -> Tensor:  # [..., R, K * D]

        *B, R, C = x.size()
        R_train = y.size(-1)
        K = self.readout_token.size(-2)
        D = self.channels
        row_key_lengths = (
            None
            if num_columns is None
            else (K + num_columns).unsqueeze(-1).to(torch.int32)
        )

        y_emb: Tensor | None = None  # [..., R_train, D]
        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))

        if torch.is_grad_enabled():
            if offload is not None:
                raise RuntimeError(
                    "'offload' is only supported when gradients are disabled"
                )
            return self._forward_autograd(
                x=x,
                y_emb=y_emb,
                categorical_mask=categorical_mask,
                train_size=R_train,
                cache=cache,
                num_columns=num_columns,
                row_key_lengths=row_key_lengths,
            )

        impute_mean = self.cell_embedding.impute_mean(
            x,
            train_size=R_train,
            cache=cache,
        )
        buffer = TokenBuffer(
            (*B, R, K + C, D),
            dtype=torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype,
            device=x.device,
            offload=offload,
        )

        def embed(tokens: Tensor, rows: slice) -> None:  # [..., N, K + C, D]
            tokens[..., :K, :] = self.readout_token.to(tokens.dtype)
            self.cell_embedding(
                x=x[..., rows, :],
                categorical_mask=categorical_mask,
                impute_mean=impute_mean,
                num_columns=num_columns,
                batch_size_limit="auto",
                out=tokens[..., K:, :],
            )
            if y_emb is not None:
                add_labels(tokens[..., K:, :], y_emb, rows)

        with buffer:
            buffer.rows(embed, load=False)
            y_emb = None

            readout = buffer.readout(K)  # [..., R, K, D]
            for i, (col_block, row_block) in enumerate(
                zip(self.col_blocks, self.row_blocks)
            ):
                run_column_layers(
                    buffer,
                    [(f"row_embedding.col_block{i}", col_block)],
                    tokens=slice(K, K + C),
                    key_rows=[slice(R_train)],
                    cache=cache,
                )

                last = i == len(self.row_blocks) - 1

                def update(
                    tokens: Tensor,  # [..., N, K + C, D]
                    rows: slice,
                    row_block: torch.nn.Module = row_block,
                    last: bool = last,
                ) -> None:
                    row_block(
                        query=tokens[..., :K, :] if last else tokens,
                        key_value=tokens,
                        seqused_key_value=row_key_lengths,
                        batch_size_limit="auto",
                        out=readout[..., rows, :, :] if last else tokens,
                    )

                buffer.rows(update, store=not last)

        return self.norm(readout).flatten(-2)

    def _forward_autograd(
        self,
        x: Tensor,  # [..., R, C]
        y_emb: Tensor | None,  # [..., R_train, D]
        categorical_mask: Tensor,  # [..., C]
        *,
        train_size: int,
        cache: Cache | None,
        num_columns: Tensor | None,
        row_key_lengths: Tensor | None,
    ) -> Tensor:  # [..., R, K * D]
        *B, R, _ = x.size()
        R_train = train_size
        K = self.readout_token.size(-2)
        D = self.channels

        x = self.cell_embedding(
            x=x,
            categorical_mask=categorical_mask,
            train_size=R_train,
            num_columns=num_columns,
            cache=cache,
        )  # [..., R, C, D]
        if y_emb is not None:
            x[..., :R_train, :, :] += y_emb.unsqueeze(-2).to(x.dtype)

        for i, (col_block, row_block) in enumerate(
            zip(self.col_blocks, self.row_blocks)
        ):
            if i > 0:
                readout_token, x = x.split([K, x.size(-2) - K], dim=-2)
                readout_token = readout_token.clone()
            else:
                readout_token = self.readout_token
                readout_token = readout_token.view(*(1,) * len(B), 1, K, D)
                readout_token = readout_token.expand(*B, R, K, D)
            x = x.transpose(-2, -3)  # [..., C, R, D]

            key = f"row_embedding.col_block{i}"
            result = col_block(
                query=x,
                key_value=cast(KVCacheEntry, cache[key])
                if cache is not None and cache.is_replaying
                else x[..., :R_train, :],
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit="auto",
            )
            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result
            del result

            x = torch.cat(
                [readout_token.to(x.dtype), x.transpose(-2, -3)],
                dim=-2,
            )
            x = row_block(
                query=x[..., :K, :] if i == len(self.row_blocks) - 1 else x,
                key_value=x,
                seqused_key_value=row_key_lengths,
                batch_size_limit="auto",
            )

        return self.norm(x).flatten(-2)
