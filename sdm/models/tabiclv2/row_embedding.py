# Copyright (c) 2025, Soda team @ Inria
# Licensed under the BSD 3-Clause License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, ModuleList, Parameter

from sdm._token_buffer import TokenBuffer, add_labels, run_column_layers
from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabiclv2.block import TabICLv2TransformerBlock
from sdm.nn import InducedTransformerBlock, RotaryEmbedding
from sdm.nn.linear import Linear


class RowEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        group_size: int,
        num_inducing_points: int,
        num_readout_tokens: int,
        norm_bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.lin = Linear(group_size, channels, **factory_kwargs)

        self.num_classes = num_classes
        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = torch.nn.Linear(1, channels, **factory_kwargs)

        self.col_layers = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_inducing_points=num_inducing_points,
                inducing_block=TabICLv2TransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    norm_bias=norm_bias,
                    qassmax=True,
                    **factory_kwargs,
                ),
                output_block=TabICLv2TransformerBlock(
                    channels=channels,
                    num_heads=num_heads,
                    norm_bias=norm_bias,
                    qassmax=False,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.readout_token = Parameter(
            torch.empty((num_readout_tokens, channels), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        rope = RotaryEmbedding(
            channels=channels // num_heads,
            layout="split_half",
            theta=100_000,
            **factory_kwargs,
        )
        self.row_layers = ModuleList(
            TabICLv2TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                norm_bias=norm_bias,
                qassmax=False,
                rope=rope,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        *,
        train_mask: Tensor | None = None,  # [R],
        max_keys: int | None = None,
        num_classes: int | None = None,
        cache: Cache | None = None,
        generator: torch.Generator | None = None,
        offload: Literal["cpu"] | None = None,
    ) -> Tensor:  # [..., R, K * D]
        if offload is not None and train_mask is not None:
            raise ValueError("'offload' does not support 'train_mask'")

        *B, R, C = x.size()
        R_train = y.size(-1)
        G, D = self.lin.in_features, self.lin.out_features
        K = self.readout_token.size(-2)

        # Feature grouping: gather G columns into each token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]

        # Context rows each column layer attends to:
        key_rows: list[Any] = []
        if cache is None or not cache.is_replaying:
            rows = slice(R_train) if train_mask is None else train_mask
            for _ in self.col_layers:
                if max_keys is None or R_train <= max_keys:
                    key_rows.append(rows)
                    continue
                perm = torch.randperm(
                    R_train,
                    device=x.device,
                    generator=generator,
                )[:max_keys]
                key_rows.append(
                    perm
                    if train_mask is None
                    else train_mask.nonzero().squeeze(-1)[perm]
                )
        else:
            key_rows = [None] * len(self.col_layers)

        num_digits = 1
        if (
            self.y_emb is not None
            and num_classes is not None
            and num_classes > self.num_classes
        ):
            bases = _mixed_radix_bases(num_classes, self.num_classes)
            num_digits = len(bases)
            if y.numel() > 0:
                y = _mixed_radix_digits(y, bases)  # [F, ..., R_train]

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
                x=x[..., index],
                y_emb=y_emb,
                train_mask=slice(R_train)
                if train_mask is None
                else train_mask,
                key_rows=key_rows,
                num_digits=num_digits,
                cache=cache,
            )

        buffer = TokenBuffer(
            ((num_digits,) if num_digits > 1 else ()) + (*B, R, K + C, D),
            dtype=torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype,
            device=x.device,
            offload=offload,
        )

        def embed(tokens: Tensor, rows: slice) -> None:  # [..., N, K + C, D]
            cells = tokens[..., K:, :]
            if num_digits > 1:
                self.lin(x[..., rows, :][..., index], out=cells[0])
                cells[1:] = cells[:1]
            else:
                self.lin(x[..., rows, :][..., index], out=cells)
            if y_emb is None:
                return
            if train_mask is None:
                add_labels(cells, y_emb, rows)
            else:
                cells[..., train_mask, :, :] += y_emb.unsqueeze(-2).to(
                    cells.dtype
                )

        with buffer:
            buffer.rows(embed, load=False)
            y_emb = None

            run_column_layers(
                buffer,
                [
                    (f"row_embedding.col_layer{i}", col_layer)
                    for i, col_layer in enumerate(self.col_layers)
                ],
                tokens=slice(K, K + C),
                key_rows=key_rows,
                cache=cache,
            )

        if num_digits > 1:  # Average over mixed-radix digits.
            buffer = TokenBuffer.from_tensor(buffer.tensor.mean(dim=0))

        with buffer:
            readout = buffer.readout(K)  # [..., R, K, D]

            def update(
                tokens: Tensor,  # [..., N, K + C, D]
                rows: slice,
            ) -> None:
                tokens[..., :K, :] = self.readout_token.to(tokens.dtype)
                for i, row_layer in enumerate(self.row_layers):
                    last = i == len(self.row_layers) - 1
                    row_layer(
                        query=tokens[..., :K, :] if last else tokens,
                        key_value=tokens,  # [..., N, K + C, D]
                        batch_size_limit="auto",
                        out=readout[..., rows, :, :] if last else tokens,
                    )

            buffer.rows(update, store=False)

        return self.norm(readout).view(*B, R, K * D)  # [..., R, K * D]

    def _forward_autograd(
        self,
        x: Tensor,  # [..., R, C, G]
        y_emb: Tensor | None,  # [..., R_train, D]
        *,
        train_mask: Any,
        key_rows: list[Any],
        num_digits: int,
        cache: Cache | None,
    ) -> Tensor:  # [..., R, K * D]
        *B, R, _, _ = x.size()
        D = self.lin.out_features
        K = self.readout_token.size(-2)

        x = self.lin(x)  # [..., R, C, D]
        if num_digits > 1:
            if y_emb is not None:
                x = x.repeat(num_digits, *(1,) * x.dim())
            else:
                x = x.expand(num_digits, *x.size())

        if y_emb is not None:
            x[..., train_mask, :, :] += y_emb.unsqueeze(-2).to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis).
        x = x.transpose(-2, -3)  # [..., C, R, D]
        for i, (col_layer, rows) in enumerate(zip(self.col_layers, key_rows)):
            key = f"row_embedding.col_layer{i}"
            result = col_layer(
                query=x,  # [..., C, R, D]
                key_value=cast(KVCacheEntry, cache[key])
                if cache is not None and cache.is_replaying
                else x[..., rows, :],  # [..., C, R_train, D]
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit="auto",
            )  # [..., C, R, D]

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result
            del result

        x = x.transpose(-2, -3)  # [..., R, C, D]

        if num_digits > 1:  # Average over mixed-radix digits.
            x = x.mean(dim=0)  # [..., R, C, D]

        # Prepend readout tokens before row-wise attention.
        readout_token = self.readout_token.to(x.dtype)
        readout_token = readout_token.view(*(1,) * len(B), 1, K, D)
        readout_token = readout_token.expand(*B, R, K, D)
        x = torch.cat([readout_token, x], dim=-2)  # [..., R, K + C, D]

        # Row-wise attention (B * R as the batch axis).
        for i, row_layer in enumerate(self.row_layers):
            x = row_layer(
                query=x[..., :K, :] if i == len(self.row_layers) - 1 else x,
                key_value=x,  # [..., R, K + C, D]
                batch_size_limit="auto",
            )  # [..., R, K + C, D] or [..., R, K, D]

        return self.norm(x).view(*B, R, K * D)  # [..., R, K * D]


def _mixed_radix_bases(num_classes: int, max_classes: int) -> list[int]:
    num_digits = math.ceil(math.log(num_classes) / math.log(max_classes))
    base = min(math.ceil(num_classes ** (1.0 / num_digits)), max_classes)
    bases = [base] * num_digits
    product = base**num_digits
    for i in range(num_digits):
        if product >= num_classes:
            break
        if bases[i] < max_classes:
            product = product // bases[i] * (bases[i] + 1)
            bases[i] += 1

    return bases


def _mixed_radix_digits(y: Tensor, bases: list[int]) -> Tensor:
    F = len(bases)
    divisors = [1] * F
    for i in range(F - 2, -1, -1):
        divisors[i] = divisors[i + 1] * bases[i + 1]

    size = (-1,) + (1,) * y.dim()
    divisor = torch.tensor(divisors, device=y.device).view(size)
    base = torch.tensor(bases, device=y.device).view(size)
    return (y.unsqueeze(0) // divisor) % base  # [F, ...]
