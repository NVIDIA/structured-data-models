# ruff: noqa: D101

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, RMSNorm, Sequential

from sdm.nn import LogScale, RotaryEmbedding, TransformerBlock


class _RMSNorm(RMSNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).to(x.dtype)


class KumoTabularTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        query_log_scale: bool,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        query_transforms: list[torch.nn.Module] = []
        key_transforms: list[torch.nn.Module] = []
        if rope is not None:
            query_transforms.append(rope)
            key_transforms.append(rope)
        query_transforms.append(
            _RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )
        key_transforms.append(
            _RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )

        mlp = Sequential(
            _RMSNorm(channels, **factory_kwargs),
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
            query_norm=_RMSNorm(channels, **factory_kwargs),
            key_value_norm=_RMSNorm(channels, **factory_kwargs),
            mlp_batch_size_divisor=2,
            query_transform=Sequential(*query_transforms),
            key_transform=Sequential(*key_transforms),
            query_scaling=LogScale(num_heads, **factory_kwargs)
            if query_log_scale
            else None,
            **factory_kwargs,
        )
