# ruff: noqa: D101

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleList, RMSNorm, Sequential

from sdm.nn import RotaryEmbedding, SoftplusScale, SwiGLU, TransformerBlock


class TabFMTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        query_transforms: list[torch.nn.Module] = [
            RMSNorm(channels // num_heads, eps=1e-6, **factory_kwargs),
            SoftplusScale(
                channels=channels // num_heads,
                multiplier=1.442695041 / math.sqrt(channels // num_heads),
                **factory_kwargs,
            ),
        ]
        key_transforms: list[torch.nn.Module] = [
            RMSNorm(channels // num_heads, eps=1e-6, **factory_kwargs),
        ]
        if rope is not None:
            query_transforms = [rope, *query_transforms]
            key_transforms = [rope, *key_transforms]

        norm = RMSNorm(channels, eps=1e-6, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=Sequential(
                RMSNorm(channels, eps=1e-6, **factory_kwargs),
                SwiGLU(
                    channels,
                    hidden_channels,
                    bias=False,
                    **factory_kwargs,
                ),
                RMSNorm(channels, eps=1e-6, **factory_kwargs),
            ),
            query_norm=norm,
            key_value_norm=norm,
            post_attn_norm=RMSNorm(channels, eps=1e-6, **factory_kwargs),
            query_transform=Sequential(*query_transforms),
            key_transform=Sequential(*key_transforms),
            scale=1.0,
            bias=False,
            **factory_kwargs,
        )


class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        hidden_channels: int,
        rope_theta: float | None = 100_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.blocks = ModuleList(
            TabFMTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                rope=None
                if rope_theta is None
                else RotaryEmbedding(
                    channels=channels // num_heads,
                    layout="interleaved",
                    theta=rope_theta,
                    requires_grad=False,
                    **factory_kwargs,
                ),
                **factory_kwargs,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        tensor: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        for block in self.blocks:
            tensor = block(query=tensor, attn_mask=attn_mask)
        return tensor
