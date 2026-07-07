# ruff: noqa: D101, D102

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
        max_train: int | None = None,
        generator: torch.Generator | None = None,
        fallback_to_all: bool = False,
    ) -> Tensor:  # [..., R, K * D]
        if max_train is not None:
            if isinstance(max_train, bool) or not isinstance(max_train, int):
                raise TypeError("`max_train` must be an integer or None")
            if max_train < 1:
                raise ValueError("`max_train` must be positive or None")

        *B, R, C = x.size()
        R_train = y.size(-1)
        G, D = self.lin.in_features, self.lin.out_features
        K = self.readout_token.size(-2)
        train_index: Any = slice(R_train)
        if train_mask is not None:
            if not isinstance(train_mask, Tensor):
                raise TypeError("`train_mask` must be a tensor")
            if train_mask.dtype != torch.bool:
                raise TypeError("`train_mask` must have boolean dtype")
            if train_mask.dim() != 1 or train_mask.size(0) != R:
                raise ValueError(
                    f"`train_mask` must be a 1D tensor with length {R}"
                )
            if train_mask.device != x.device:
                raise ValueError(
                    "`train_mask` must be on the same device as `x`"
                )
            if train_mask.count_nonzero() != R_train:
                raise ValueError(
                    f"`train_mask` must select exactly {R_train} rows to "
                    "match `y`"
                )
            train_index = train_mask

        if self.y_emb is not None:
            if y.is_floating_point() or y.is_complex():
                raise TypeError(
                    "Classification targets must have an integral or "
                    "boolean dtype"
                )
            y = y.long()
        else:
            if not y.is_floating_point():
                raise TypeError(
                    "Regression targets must have a floating-point dtype"
                )
            y = y.to(dtype=x.dtype)

        # Feature grouping: gather G columns into each token.
        shift = 2 ** torch.arange(G, device=x.device)
        index = torch.arange(C, device=x.device)
        index = (index.view(C, 1) + shift.view(1, G)) % C  # [C, G]
        x = x[..., index]  # [..., R, C, G]
        x = self.lin(x)  # [..., R, C, D]

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y).view(*B, R_train, 1, D)
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1)).view(*B, R_train, 1, D)

            x[..., train_index, :, :] += y_emb.to(x.dtype)

        # Column-wise induced set attention (B * C as the batch axis):
        x = x.transpose(-2, -3)  # [..., C, R, D]
        for i, col_layer in enumerate(self.col_layers):
            key = f"row_embedding.col_layer{i}"
            if cache is not None and cache.is_replaying:
                key_value = cache[key]
            else:
                key_value = x[..., train_index, :]
                if key_value.size(-2) == 0:
                    if not fallback_to_all:
                        raise ValueError(
                            "Column-attention context is empty; pass "
                            "`fallback_to_all=True` to use all local rows"
                        )
                    if cache is not None and cache.is_recording:
                        raise ValueError(
                            "Cannot record a cache from all-row fallback "
                            "context"
                        )
                    key_value = x
                if max_train is not None and key_value.size(-2) > max_train:
                    if (
                        generator is not None
                        and generator.device != key_value.device
                    ):
                        raise ValueError(
                            "`generator` must be on the same device as the "
                            "column-attention context"
                        )
                    index = torch.randperm(
                        key_value.size(-2),
                        device=key_value.device,
                        generator=generator,
                    )[:max_train]
                    key_value = key_value.index_select(-2, index)

            result = col_layer(
                query=x,  # [..., C, R, D]
                key_value=key_value,  # [..., C, R_context, D]
                return_key_value=cache is not None and cache.is_recording,
            )  # [..., C, R, D]

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

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
