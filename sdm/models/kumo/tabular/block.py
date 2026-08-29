from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, RMSNorm, Sequential

from sdm.nn import LogScale, RotaryEmbedding, TransformerBlock


class KumoTabularTransformerBlock(TransformerBlock):
    """Transformer block used by Kumo Tabular.

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        rope: Optional rotary embedding applied to query and key heads.
        query_log_scale: Whether to scale queries by a learned factor and the
            logarithm of the key length.
        device: Device of the parameters.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        rope: RotaryEmbedding | None = None,
        query_log_scale: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        head_channels = channels // num_heads

        query_transforms: list[torch.nn.Module] = []
        key_transforms: list[torch.nn.Module] = []
        if rope is not None:
            query_transforms.append(rope)
            key_transforms.append(rope)
        query_transforms.append(
            RMSNorm(
                head_channels,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )
        key_transforms.append(
            RMSNorm(
                head_channels,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )

        mlp_out = Linear(2 * channels, channels, **factory_kwargs)
        torch.nn.init.zeros_(mlp_out.weight)
        torch.nn.init.zeros_(cast(Tensor, mlp_out.bias))
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=Sequential(
                RMSNorm(channels, **factory_kwargs),
                Linear(channels, 2 * channels, **factory_kwargs),
                GELU(),
                mlp_out,
            ),
            query_norm=RMSNorm(channels, **factory_kwargs),
            key_value_norm=RMSNorm(channels, **factory_kwargs),
            query_transform=Sequential(*query_transforms),
            key_transform=Sequential(*key_transforms),
            query_scaling=LogScale(num_heads, **factory_kwargs)
            if query_log_scale
            else None,
            **factory_kwargs,
        )
