"""Feed-forward modules for structured tensor models."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


class SwiGLUFeedForward(torch.nn.Module):
    r"""Feed-forward projection with a SwiGLU hidden layer.

    Args:
        channels: The number of input and output channels.
        feedforward_channels: The hidden width.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        feedforward_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.value_lin = Linear(
            channels,
            feedforward_channels,
            **factory_kwargs,
        )
        self.gate_lin = Linear(
            channels,
            feedforward_channels,
            **factory_kwargs,
        )
        self.out_lin = Linear(
            feedforward_channels,
            channels,
            **factory_kwargs,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Apply the SwiGLU feed-forward projection."""
        value = self.value_lin(tensor)
        gate = F.silu(self.gate_lin(tensor))
        return self.out_lin(gate * value)


class ChunkedFeedForward(torch.nn.Module):
    r"""Apply a tokenwise feed-forward module in bounded-size chunks.

    The wrapped module must independently transform the final dimension of
    each input token. Keeping this wrapper present when chunking is disabled
    preserves child parameter paths across chunk-size configurations.

    Args:
        module: The tokenwise feed-forward module.
        chunk_size: The maximum number of flattened tokens per call. ``None``
            disables chunking.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        chunk_size: int | None,
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_size = chunk_size

    def forward(self, tensor: Tensor) -> Tensor:
        r"""Apply the wrapped feed-forward module."""
        if self.chunk_size is None:
            return self.module(tensor)

        batch_shape = tensor.shape[:-1]
        tensor = tensor.reshape(-1, tensor.size(-1))
        if tensor.size(0) == 0:
            output = self.module(tensor)
        else:
            output = torch.cat(
                [
                    self.module(chunk)
                    for chunk in tensor.split(self.chunk_size)
                ],
                dim=0,
            )
        return output.reshape(*batch_shape, output.size(-1))
