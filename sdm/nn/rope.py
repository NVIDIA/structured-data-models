from typing import Any, Literal

import torch
from torch import Tensor


class RotaryEmbedding(torch.nn.Module):
    r"""RoPE from the `"RoFormer" <https://arxiv.org/abs/2104.09864>`_ paper.

    Args:
        channels: The number of channels per attention head.
        layout: The channel pairing layout. ``"split_half"`` pairs the first
            half of the channels with the second half. ``"interleaved"`` pairs
            adjacent even and odd channels.
        theta: The base frequency used to initialize inverse frequencies.
        requires_grad: Whether inverse frequencies are learnable.
        device: The device.
        dtype: The dtype.
        rotary_channels: The number of leading channels to rotate. ``None``
            rotates all channels. Remaining channels pass through unchanged.
    """

    def __init__(
        self,
        channels: int,
        layout: Literal["split_half", "interleaved"],
        theta: float = 100_000,
        requires_grad: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        rotary_channels: int | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.layout = layout
        self.channels = channels
        self.rotary_channels = (
            channels if rotary_channels is None else rotary_channels
        )

        if channels % 2 != 0:
            raise ValueError(f"'channels' must be even (got {channels})")
        if (
            self.rotary_channels < 2
            or self.rotary_channels > channels
            or self.rotary_channels % 2 != 0
        ):
            raise ValueError(
                "'rotary_channels' must be an even number between 2 and "
                f"'channels' (got {self.rotary_channels})"
            )

        arange = torch.arange(0, self.rotary_channels, 2, **factory_kwargs)
        inv_freq = 1.0 / (theta ** (arange / self.rotary_channels))
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
        if x.size(-1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels (got {x.size(-1)})"
            )
        seq = torch.arange(x.size(-3), device=x.device, dtype=torch.float32)
        freq = seq.view(-1, 1) * self.inv_freq.view(1, -1)
        # [S, C_rotary // 2]
        sin = freq.sin()[:, None, :].to(x.dtype)
        cos = freq.cos()[:, None, :].to(x.dtype)

        rotary = x[..., : self.rotary_channels]
        if self.layout == "interleaved":
            x1, x2 = rotary[..., 0::2], rotary[..., 1::2]
        else:
            assert self.layout == "split_half"
            x1, x2 = rotary.split(sin.size(-1), dim=-1)

        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        if self.layout == "interleaved":
            rotary = torch.stack((out1, out2), dim=-1).flatten(-2)
        else:
            rotary = torch.cat((out1, out2), dim=-1)

        if self.rotary_channels == self.channels:
            return rotary
        return torch.cat((rotary, x[..., self.rotary_channels :]), dim=-1)
