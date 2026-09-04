# ruff: noqa: D101

from typing import Any, cast

import torch
from torch.nn import GELU, LayerNorm, Linear, Sequential

from sdm.nn import QASSMax, RotaryEmbedding, TransformerBlock


class TabICLv2TransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        norm_bias: bool,
        qassmax: bool,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)
        mlp = Sequential(
            LayerNorm(channels, bias=norm_bias, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, channels, **factory_kwargs),
        )
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).bias)

        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=mlp,
            query_norm=norm,
            key_value_norm=norm,
            query_scaling=QASSMax(
                channels=channels // num_heads,
                num_heads=num_heads,
                **factory_kwargs,
            )
            if qassmax
            else None,
            query_transform=rope,
            key_transform=rope,
            **factory_kwargs,
        )

    def peak_bytes_per_example(
        self,
        query_length: int,
        key_value_length: int | None = None,
        *,
        dtype: torch.dtype,
    ) -> int:
        pass
