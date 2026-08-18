# ruff: noqa: D101

import math
from typing import Any

import torch
from torch.nn import RMSNorm, Sequential

from sdm.nn import RotaryEmbedding, SoftplusScale, SwiGLU, TransformerBlock


class TabFMTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
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
                SwiGLU(channels, 4 * channels, bias=False, **factory_kwargs),
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
