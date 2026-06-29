from typing import Any

import torch
from torch import Tensor


class RotaryEmbedding(torch.nn.Module):
    """Rotary Positional Embeddings (RoPE).

    Uses a split-half channel layout, pairing the first half channels with the
    last half, rather than interleaved even/odd pairs.

    Args:
        channels: The number of channels per attention head.
        theta: The base frequency used to initialize inverse frequencies.
        requires_grad: Whether inverse frequencies are learnable.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        theta: float = 100_000,
        requires_grad: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("`channels` must be even")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        arange = torch.arange(0, channels, 2, **factory_kwargs)
        inv_freq = 1.0 / (theta ** (arange / channels))
        self.inv_freq = torch.nn.Parameter(
            inv_freq,
            requires_grad=requires_grad,
        )

    def forward(
        self,
        x: Tensor,  # [..., S, H, C]
    ) -> Tensor:  # [..., S, H, C]
        """The forward pass.

        Args:
            x: Tensor with shape ``[..., S, H, C]``.
                ``S`` is the query sequence length, ``H`` is the number of
                attention heads, and ``C`` is the channels per head.

        Returns:
            Tensor with shape ``[..., S, H, C]``.
        """
        if x.size(-1) != 2 * self.inv_freq.size(-1):
            raise ValueError(
                f"Expected {2 * self.inv_freq.size(-1)} channels, "
                f"got {x.size(-1)}",
            )

        seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * self.inv_freq.view(1, -1)  # [S, C // 2]
        sin = freq.sin()[:, None, :].to(x.dtype)  # [S, 1, C // 2]
        cos = freq.cos()[:, None, :].to(x.dtype)  # [S, 1, C // 2]

        return torch.cat(
            [
                x[..., : cos.size(-1)] * cos - x[..., sin.size(-1) :] * sin,
                x[..., cos.size(-1) :] * cos + x[..., : sin.size(-1)] * sin,
            ],
            dim=-1,
        )
