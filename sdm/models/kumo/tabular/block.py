# ruff: noqa: D101

from typing import Any, cast

import torch
from torch.nn import GELU, Linear, RMSNorm, Sequential

from sdm.nn import LogScale, RotaryEmbedding, TransformerBlock


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
            RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )
        key_transforms.append(
            RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )

        mlp = Sequential(
            RMSNorm(channels, **factory_kwargs),
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
            query_norm=RMSNorm(channels, **factory_kwargs),
            key_value_norm=RMSNorm(channels, **factory_kwargs),
            query_transform=Sequential(*query_transforms),
            key_transform=Sequential(*key_transforms),
            query_scaling=LogScale(num_heads, **factory_kwargs)
            if query_log_scale
            else None,
            **factory_kwargs,
        )

    def peak_bytes_per_example(
        self,
        query_length: int,
        key_value_length: int | None = None,
        *,
        dtype: torch.dtype,
    ) -> int:
        key_value_length = (
            query_length if key_value_length is None else key_value_length
        )
        length = max(query_length, key_value_length)
        precision = torch.empty((), dtype=dtype).element_size()
        factor = 30 if precision <= 2 else 32  # TODO High peak!
        return factor * length * self.attn.q_dim
