from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList


class MLP(torch.nn.Module):
    """Apply the TabFM v1.0.0 tanh-GELU multilayer perceptron.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of each hidden layer.
        out_channels: Number of output channels.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        channels = (in_channels, *hidden_channels, out_channels)
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.layers = ModuleList(
            [
                Linear(channels[index], channels[index + 1], **factory_kwargs)
                for index in range(len(channels) - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``[..., in_channels]`` into ``[..., out_channels]``."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = F.gelu(x, approximate="tanh")
        return x
