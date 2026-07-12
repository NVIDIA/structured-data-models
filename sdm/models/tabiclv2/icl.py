# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList

from sdm.cache import Cache
from sdm.nn import TransformerBlock


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        norm_bias: bool,
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
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = TransformerBlock(
                channels=channels,
                num_query_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=True,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            self.layers.append(layer)

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        seqused_train: Tensor | None = None,  # [...]
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, D]
        R_train = y.size(-1)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))  # [..., R_train, D]

            x[..., :R_train, :] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            key = f"icl_block.layer{i}"
            result = layer(
                query=x[..., R_train:, :] if i == len(self.layers) - 1 else x,
                key_value=cache[key]
                if cache is not None and cache.is_replaying
                else x[..., :R_train, :],  # [..., R_train, D]
                seqused_key_value=seqused_train,
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit=batch_size_limit,
            )

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        return self.norm(x)  # [..., R_test, D]
