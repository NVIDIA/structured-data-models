# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList

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
                num_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=True,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            self.layers.append(layer)

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [B, R, D]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, D]
        R_train = y.size(-1)

        if self.y_emb is not None:
            y_emb = self.y_emb(y)  # [B, R_train, D]
        else:
            assert self.y_lin is not None
            y_emb = self.y_lin(y.unsqueeze(-1))  # [B, R_train, D]

        x[:, :R_train] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            x = layer(
                query=x[:, R_train:] if i == len(self.layers) - 1 else x,
                key_value=x[:, :R_train],
            )

        return self.norm(x)  # [B, R_test, D]
