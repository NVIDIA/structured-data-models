from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList


def _gelu(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


class MLP(torch.nn.Module):
    """Apply the TabFM v1.0.0 multilayer perceptron.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Width of each hidden layer.
        out_channels: Number of output channels.
        activation: Activation applied between linear layers.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Sequence[int],
        out_channels: int,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        channels = (in_channels, *hidden_channels, out_channels)
        self.layers = ModuleList(
            Linear(in_channels, out_channels, **factory_kwargs)
            for in_channels, out_channels in zip(channels, channels[1:])
        )
        self.act: Callable[[Tensor], Tensor]
        if activation == "relu":
            self.act = F.relu
        elif activation == "gelu":
            self.act = _gelu
        elif activation == "silu":
            self.act = F.silu
        else:
            assert False

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP.

        Args:
            x: Input tensor with shape ``[..., C]``. ``C`` is
                ``in_channels``.

        Returns:
            Tensor with shape ``[..., O]``. ``O`` is ``out_channels``.
        """
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)
