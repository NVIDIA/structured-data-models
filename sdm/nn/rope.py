from typing import Any, Literal

import torch
from torch import Tensor


def apply_rotary_embedding(
    input: Tensor,
    frequencies: Tensor,
    *,
    layout: Literal["split_half", "interleaved"] = "split_half",
) -> Tensor:
    """Apply rotary position embeddings along the sequence dimension.

    Args:
        input: Query or key tensor with shape ``[..., S, H, C]``. ``S`` is
            sequence length, ``H`` is the number of heads, and ``C`` is the
            number of channels per head.
        frequencies: Inverse frequencies with shape ``[C // 2]``.
        layout: Channel-pairing convention. ``"split_half"`` pairs the first
            and second channel halves; ``"interleaved"`` pairs adjacent even
            and odd channels.

    Returns:
        Tensor with shape ``[..., S, H, C]`` and the input dtype.
    """
    if input.size(-1) != 2 * frequencies.numel():
        raise ValueError(
            f"Expected {2 * frequencies.numel()} channels, "
            f"got {input.size(-1)}"
        )
    if layout not in ("split_half", "interleaved"):
        raise ValueError(f"Unsupported rotary layout: {layout!r}")

    sequence_length = input.size(-3)
    positions = torch.arange(
        sequence_length,
        device=input.device,
        dtype=torch.float32,
    )
    angles = torch.outer(positions, frequencies.float())
    sin = angles.sin().to(input.dtype)
    cos = angles.cos().to(input.dtype)

    if layout == "interleaved":
        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)
        broadcast_shape = (1,) * (input.dim() - 3) + (
            sequence_length,
            1,
            input.size(-1),
        )
        sin = sin.view(broadcast_shape)
        cos = cos.view(broadcast_shape)
        even, odd = input[..., 0::2], input[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).reshape_as(input)
        return input * cos + rotated * sin

    broadcast_shape = (1,) * (input.dim() - 3) + (
        sequence_length,
        1,
        frequencies.numel(),
    )
    sin = sin.view(broadcast_shape)
    cos = cos.view(broadcast_shape)
    first, second = input.chunk(2, dim=-1)
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin],
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
            x,
            self.inv_freq,
            layout="split_half",
        )
