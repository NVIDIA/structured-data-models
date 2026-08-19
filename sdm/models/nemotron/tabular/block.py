import math
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import RMSNorm, Sequential

from sdm.nn import RotaryEmbedding, SoftplusScale, SwiGLU, TransformerBlock


class _TransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        rope: RotaryEmbedding | None = None,
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
        query_transforms.extend(
            [
                RMSNorm(head_channels, eps=1e-6, **factory_kwargs),
                SoftplusScale(
                    channels=head_channels,
                    multiplier=1.442695041 / math.sqrt(head_channels),
                    **factory_kwargs,
                ),
            ]
        )
        key_transforms.append(
            RMSNorm(head_channels, eps=1e-6, **factory_kwargs)
        )

        post_attn_norm = RMSNorm(channels, **factory_kwargs)
        post_mlp_norm = RMSNorm(channels, **factory_kwargs)
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=Sequential(
                RMSNorm(channels, **factory_kwargs),
                SwiGLU(
                    channels=channels,
                    hidden_channels=2 * channels,
                    bias=False,
                    **factory_kwargs,
                ),
                post_mlp_norm,
            ),
            query_norm=RMSNorm(channels, **factory_kwargs),
            key_value_norm=RMSNorm(channels, **factory_kwargs),
            post_attn_norm=post_attn_norm,
            query_transform=Sequential(*query_transforms),
            key_transform=Sequential(*key_transforms),
            scale=1.0,
            **factory_kwargs,
        )

        self.attn.out_lin.reset_parameters()
        torch.nn.init.zeros_(cast(Tensor, post_attn_norm.weight))
        torch.nn.init.zeros_(cast(Tensor, post_mlp_norm.weight))
