from typing import Any, Literal

import torch
from torch import Tensor


def apply_rotary_embedding(
    x: Tensor,  # [..., S, H, C]
    inv_freq: Tensor,  # [C // 2]
    layout: Literal["split_half", "interleaved"] = "split_half",
) -> Tensor:  # [..., S, H, C]
    """Apply Rotary Position Embedding from the `RoFormer`_ paper.

    .. _RoFormer: https://arxiv.org/abs/2104.09864

    Args:
        x: Input tensor with shape ``[..., S, H, C]``. ``S`` is the sequence
            length, ``H`` is the number of attention heads, and ``C`` is the
            channels per head.
        inv_freq: Inverse frequencies with shape ``[C // 2]``.
        layout: Channel-pairing layout. ``"split_half"`` pairs the first and
            second channel halves; ``"interleaved"`` pairs adjacent channels.

    Returns:
        Tensor with shape ``[..., S, H, C]``.
    """
    if inv_freq.dim() != 1:
        raise ValueError("`inv_freq` must be one-dimensional")
    if x.size(-1) != 2 * inv_freq.size(-1):
        raise ValueError(
            f"Expected {2 * inv_freq.size(-1)} channels, got {x.size(-1)}"
        )
    if layout not in ("split_half", "interleaved"):
        raise ValueError(f"Unsupported rotary layout: {layout}")

    seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
    freq = seq.view(-1, 1) * inv_freq.view(1, -1)  # [S, C // 2]
    sin = freq.sin()[:, None, :].to(x.dtype)  # [S, 1, C // 2]
    cos = freq.cos()[:, None, :].to(x.dtype)  # [S, 1, C // 2]

    if layout == "interleaved":
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack(
            [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin],
            dim=-1,
        ).flatten(-2)

    return torch.cat(
        [
            x[..., : cos.size(-1)] * cos - x[..., sin.size(-1) :] * sin,
            x[..., cos.size(-1) :] * cos + x[..., : sin.size(-1)] * sin,
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """Rotary Positional Embedding (RoPE).

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
        return apply_rotary_embedding(
            x=x,
            inv_freq=self.inv_freq,
            layout="split_half",
        )
