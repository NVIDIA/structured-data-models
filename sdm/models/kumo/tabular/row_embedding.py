# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Parameter

from sdm._memory import chunk_memory_limit
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

        self.col_blocks: ModuleList[InducedTransformerBlock] = ModuleList(
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
        self.row_blocks: ModuleList[KumoTabularTransformerBlock] = ModuleList(
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
    ) -> Tensor:  # [..., R, K * D]
        starts = self._pass_starts(x, train_size=y.size(-1), cache=cache)
        if len(starts) == 1:
            return self._forward(x, y, categorical_mask, cache=cache)

        # Query rows only read context state, which the first pass records
        # for replay in later passes.
        cache = Cache() if cache is None else cache
        first = self._forward(
            x=x[..., : starts[1], :],
            y=y,
            categorical_mask=categorical_mask,
            cache=cache,
        )  # [..., starts[1], K * D]
        cache.freeze()
        out = first.new_empty((*first.shape[:-2], x.size(-2), first.size(-1)))
        out[..., : starts[1], :] = first
        del first
        ends = [*starts[2:], x.size(-2)]
        for start, end in zip(starts[1:], ends, strict=True):
            out[..., start:end, :] = self._forward(
                x=x[..., start:end, :],
                y=y[..., :0],
                categorical_mask=categorical_mask,
                cache=cache,
            )
        return out

    def _pass_starts(
        self,
        x: Tensor,  # [..., R, C]
        train_size: int,
        cache: Cache | None,
    ) -> list[int]:
        # First rows of the passes that embed the rows of `x`. Passes run
        # without gradients on CUDA and replay context state from a cache.
        if (
            torch.is_grad_enabled()
            or not x.is_cuda
            or (cache is not None and cache.is_recording)
        ):
            return [0]
        *B, R, C = x.size()
        N = math.prod(B)
        K, D = self.readout_token.size(-2), self.channels
        G = self.cell_embedding.group_size
        M = self.col_blocks[0].inducing_points.size(-2)
        s = (
            torch.get_autocast_dtype(x.device.type).itemsize
            if torch.is_autocast_enabled(x.device.type)
            else x.element_size()
        )
        budget = chunk_memory_limit(x.device)
        # Bytes per row: the cell buffer, plus the missingness mask, imputed
        # values and their feature groups while embedding cells.
        row_bytes = N * (
            (K + C) * D * s + (G + 1) * (x.element_size() + 1) * C
        )
        # Without a cache, the context pass records the key/value projections
        # of all column blocks for the query passes. Query rows that fit the
        # chunk memory budget plus these projections run with the context.
        state_bytes = 0
        if cache is None:
            state_bytes = 2 * N * C * M * D * s * len(self.col_blocks)
        if (R - train_size) * row_bytes <= budget + state_bytes:
            return [0]

        # Row blocks run the rows of all batch entries in chunks of `chunk`
        # rows. In passes starting on multiples of `grid` rows, every row runs
        # in a chunk of the same size as in a single pass, since the last pass
        # holds the partial last chunk of a single pass, the last
        # `N * R % chunk` rows of the last batch entry. Attention over long
        # rows rounds differently in small chunks, so this keeps passes equal
        # to a single pass up to rare rounding differences in small passes.
        chunk = self.row_blocks[0].auto_batch_size_limit(
            device=x.device,
            element_size=s,
            query_length=K + C,
            key_value_length=K + C,
        )
        grid = chunk // math.gcd(N, chunk)
        context = -(-train_size // grid) * grid
        last = R - max(N * R % chunk, 1)
        if context > last:
            return [0]
        # Balanced query passes need no more memory than the context pass,
        # the budget or one grid of rows, whichever is more.
        grids = max(max(train_size, budget // row_bytes) // grid, 1)
        num_passes = -(-(R - context) // (grids * grid))
        step = -(-(R - context) // (num_passes * grid)) * grid
        return [0, *range(context or step, last + 1, step)]

    def _forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R, K * D]

        *B, R, C = x.size()
        R_train = y.size(-1)
        K = self.readout_token.size(-2)
        D = self.channels

        buffer: Tensor | None = None
        if torch.is_grad_enabled():
            x = self.cell_embedding(
                x=x,
                categorical_mask=categorical_mask,
                train_size=R_train,
                cache=cache,
            )  # [..., R, C, D]
        else:
            buffer = torch.empty(
                (*B, R, K + C, D),
                device=x.device,
                dtype=torch.get_autocast_dtype(x.device.type)
                if torch.is_autocast_enabled(x.device.type)
                else x.dtype,
            )
            buffer[..., :K, :] = self.readout_token.to(buffer.dtype)
            x = self.cell_embedding(
                x=x,
                categorical_mask=categorical_mask,
                train_size=R_train,
                cache=cache,
                batch_size_limit="auto",
                out=buffer[..., K:, :],
            )

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)  # [..., R_train, 1, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)
            x[..., :R_train, :, :] += y_emb.to(x.dtype)

        if buffer is not None:
            x = x.transpose(-2, -3)  # [..., C, R, D]

        for i, (col_block, row_block) in enumerate(
            zip(self.col_blocks, self.row_blocks)
        ):
            if buffer is None:
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
                out=None if buffer is None else x,
            )

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result
            del result

            if buffer is None:
                x = torch.cat(
                    [readout_token.to(x.dtype), x.transpose(-2, -3)],
                    dim=-2,
                )
                x = row_block(
                    query=x[..., :K, :]
                    if i == len(self.row_blocks) - 1
                    else x,
                    key_value=x,
                    batch_size_limit="auto",
                )
            else:
                buffer = row_block(
                    query=buffer[..., :K, :]
                    if i == len(self.row_blocks) - 1
                    else buffer,
                    key_value=buffer,
                    batch_size_limit="auto",
                    out=buffer[..., :K, :]
                    if i == len(self.row_blocks) - 1
                    else buffer,
                )

        return self.norm(buffer if buffer is not None else x).flatten(-2)
