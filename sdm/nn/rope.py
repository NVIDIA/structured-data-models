from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor


class RotaryEmbedding(torch.nn.Module):
    """Rotary Positional Embedding (RoPE).

    Args:
        channels: The number of channels per attention head.
        layout: The channel pairing layout. ``""split_half"`` pairs the first
            half of the channels with the second half. ``"interleaved"`` pairs
            adjacent even and off channels.
        theta: The base frequency used to initialize inverse frequencies.
        requires_grad: Whether inverse frequencies are learnable.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        layout: Literal["split_half", "interleaved"],
        theta: float = 100_000,
        requires_grad: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.layout = layout

        if channels % 2 != 0:
            raise ValueError(f"'channels' must be even (got {channels})")

        arange = torch.arange(0, channels, 2, **factory_kwargs)
        inv_freq = 1.0 / (theta ** (arange / channels))
        self.inv_freq = torch.nn.Parameter(
            inv_freq,
            requires_grad=requires_grad,
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "RotaryEmbedding":
        # Rotary phases are precision-critical: keep the inverse frequencies
        # in their construction dtype (float32 by default) through casts such
        # as `module.to(torch.bfloat16)` (device moves still apply). The
        # original values are restored rather than upcast, since the cast
        # itself already rounds them. The forward pass type-promotes as
        # needed and casts sin/cos to the input dtype.
        original = self.inv_freq.detach().clone()
        out = super()._apply(fn, recurse)
        if self.inv_freq.dtype != original.dtype:
            self.inv_freq.data = original.to(device=self.inv_freq.device)
        return out

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
                f"Expected {2 * self.inv_freq.size(-1)} channels "
                f"(got {x.size(-1)})"
            )
        seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * self.inv_freq.view(1, -1)  # [S, C // 2]
        sin = freq.sin()[:, None, :].to(x.dtype)  # [S, 1, C // 2]
        cos = freq.cos()[:, None, :].to(x.dtype)  # [S, 1, C // 2]

        if self.layout == "interleaved":
            x1, x2 = x[..., 0::2], x[..., 1::2]
        else:
            assert self.layout == "split_half"
            x1, x2 = x.split(sin.size(-1), dim=-1)

        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        if self.layout == "interleaved":
            return torch.stack((out1, out2), dim=-1).flatten(-2)
        return torch.cat((out1, out2), dim=-1)
