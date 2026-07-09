# ruff: noqa: D101, D102

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList, Parameter

from sdm.cache import Cache
from sdm.nn import InducedTransformerBlock, RotaryEmbedding, TransformerBlock


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

        self.max_classes = num_classes
        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.col_layers = ModuleList(
            InducedTransformerBlock(
                channels=channels,
                num_query_heads=num_heads,
                feedforward_channels=2 * channels,
                num_inducing_points=num_inducing_points,
                qassmax=True,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.readout_token = Parameter(
            torch.empty((num_readout_tokens, channels), **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.row_layers = ModuleList(
            TransformerBlock(
                channels=channels,
                num_query_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=False,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.rope = RotaryEmbedding(
            channels=channels // num_heads,
            theta=100_000,
            **factory_kwargs,
        )

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        *,
        train_mask: Tensor | None = None,  # [R],
        cache: Cache | None = None,
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
        x = self.lin(x)  # [..., R, C, D]

        num_digits = 1
        if y.numel() > 0:
            if self.y_emb is not None:
                # TODO Cache `num_classes` to avoid device synchronization.
                num_classes = int(y.max()) + 1
                if num_classes > self.max_classes:
                    # TODO Support KV cache
                    if cache is not None:
                        raise NotImplementedError(
                            f"Key/value caching is not supported with more "
                            f"than {self.max_classes} classes "
                            f"(got {num_classes})"
                        )

                    bases = _mixed_radix_bases(num_classes, self.max_classes)
                    num_digits = len(bases)
                    y = _mixed_radix_digits(y, bases)  # [F, ..., R_train]
                    x = x.unsqueeze(0).repeat(num_digits, *(1,) * x.dim())
                y_emb = self.y_emb(y).unsqueeze(-2)
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).unsqueeze(-2)

            # y_emb has shape [F, ..., R_train, 1, D]:
            x[..., train_mask, :, :] += y_emb.to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis):
        x = x.transpose(-2, -3)  # [..., C, R, D]
        for i, col_layer in enumerate(self.col_layers):
            key = f"row_embedding.col_layer{i}"
            result = col_layer(
                query=x,  # [..., C, R, D]
                key_value=cache[key]
                if cache is not None and cache.is_replaying
                else x[..., train_mask, :],  # [..., C, R_train, D],
                return_key_value=cache is not None and cache.is_recording,
            )  # [..., C, R, D]

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        if num_digits > 1:  # Average over mixed-radix digits.
            x = x.mean(dim=0)  # [F, ..., C, R, D] -> [..., C, R, D]

        x = torch.cat(  # Prepend readout tokens before row-wise attention.
            [
                self.readout_token.to(x.dtype)
                .view(*(1,) * len(B), 1, K, D)
                .expand(*B, R, K, D),
                x.transpose(-2, -3),  # [..., R, C, D]
            ],
            dim=-2,
        )  # [..., R, K + C, D]

        # Row-wise attention (B * R as the batch axis).
        for i, row_layer in enumerate(self.row_layers):
            x = row_layer(
                query=x[..., :K, :] if i == len(self.row_layers) - 1 else x,
                key_value=x,  # [..., R, K + C, D]
                rope=self.rope,
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
