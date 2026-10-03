# Copyright (c) 2025, Soda team @ Inria
# Licensed under the BSD 3-Clause License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, ModuleList, Parameter

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
    ) -> Tensor:  # [..., R, K * D]
        *B, R, C = x.size()
        R_train = y.size(-1)
        G, D = self.lin.in_features, self.lin.out_features
        K = self.readout_token.size(-2)
        train_mask: Any = slice(R_train) if train_mask is None else train_mask

        # Feature grouping: gather G columns into each token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]
        x = x[..., index]  # [..., R, C, G]

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

        buffer: Tensor | None = None
        if torch.is_grad_enabled():
            x = self.lin(x)  # [..., R, C, D]
            if num_digits > 1:
                if y.numel() > 0:
                    x = x.repeat(num_digits, *(1,) * x.dim())
                else:
                    x = x.expand(num_digits, *x.size())
        else:
            buffer = torch.empty(
                ((num_digits,) if num_digits > 1 else ())
                + (*x.size()[:-3], R, K + C, D),
                device=x.device,
                dtype=torch.get_autocast_dtype(x.device.type)
                if torch.is_autocast_enabled(x.device.type)
                else x.dtype,
            )
            if num_digits > 1:
                self.lin(x, out=buffer[0, ..., K:, :])
                buffer[1:, ..., K:, :] = buffer[:1, ..., K:, :]
                x = buffer[..., K:, :]
            else:
                x = self.lin(x, out=buffer[..., K:, :])  # [..., R, C, D]

        projection_dtype = x.dtype
        stabilize_amp = (
            y.numel() > 0
            and x.is_cuda
            and projection_dtype in (torch.float16, torch.bfloat16)
            and self.lin.weight.dtype == torch.float32
        )
        if stabilize_amp:
            # Preserve the residual carrier before target injection.
            if buffer is None:
                x = x.float()
            else:
                buffer = buffer.float()
                x = buffer[..., K:, :]

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).unsqueeze(-2)  # [..., R_train, 1, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)

            x[..., train_mask, :, :] += y_emb.to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis).
        x = x.transpose(-2, -3)  # [..., C, R, D]
        for i, col_layer in enumerate(self.col_layers):
            key = f"row_embedding.col_layer{i}"
            if cache is not None and cache.is_replaying:
                key_value = cast(KVCacheEntry, cache[key])
            else:
                key_value = x[..., train_mask, :]
                if max_keys is not None and key_value.size(-2) > max_keys:
                    index = torch.randperm(
                        key_value.size(-2),
                        device=key_value.device,
                        generator=generator,
                    )[:max_keys]
                    key_value = key_value[..., index, :]

            result = col_layer(
                query=x,  # [..., C, R, D]
                key_value=key_value,  # [..., C, R_train, D]
                return_key_value=cache is not None and cache.is_recording,
                # The target-conditioned column blocks are AMP-sensitive.
                _force_output_block_float32=stabilize_amp,
                batch_size_limit="auto",
                out=None if torch.is_grad_enabled() else x,
            )  # [..., C, R, D]
            del key_value

            if cache is not None and cache.is_recording:
                x, key_value = result
                if stabilize_amp:
                    # Replay uses the original AMP dtype.
                    key_value = KVCacheEntry(
                        key=key_value.key.to(dtype=projection_dtype),
                        value=key_value.value.to(dtype=projection_dtype),
                    )
                cache[key] = key_value
            else:
                x = result
            del result

        x = x.transpose(-2, -3)  # [..., R, C, D]

        if num_digits > 1:  # Average over mixed-radix digits.
            if buffer is not None:
                buffer = buffer.mean(dim=0)  #  [..., R, K + C, D]
            else:
                x = x.mean(dim=0)  # [..., R, C, D]

        # Prepend readout tokens before row-wise attention.
        readout_token = self.readout_token.to(x.dtype)
        readout_token = readout_token.view(*(1,) * len(B), 1, K, D)
        readout_token = readout_token.expand(*B, R, K, D)

        if buffer is not None:
            buffer[..., :, :K, :] = readout_token
            x = buffer
        else:
            x = torch.cat([readout_token, x], dim=-2)  # [..., R, K + C, D]

        # Row-wise attention (B * R as the batch axis).
        for i, row_layer in enumerate(self.row_layers):
            x = row_layer(
                query=x[..., :K, :] if i == len(self.row_layers) - 1 else x,
                key_value=x,  # [..., R, K + C, D]
                batch_size_limit="auto",
                out=None
                if torch.is_grad_enabled()
                else x[..., :K, :]
                if i == len(self.row_layers) - 1
                else x,
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
