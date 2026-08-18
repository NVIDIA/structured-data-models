# ruff: noqa: D205

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


class SwiGLU(torch.nn.Module):
    r""":math:`\mathrm{FFN}_{\mathrm{SwiGLU}}` block from the `"GLU Variants
    Improve Transformer" <https://arxiv.org/abs/2002.05202>`_ paper.

    .. math::

        W_{\mathrm{down}}(
            \operatorname{Swish}_1(W_{\mathrm{gate}} x + b_{\mathrm{gate}})
            \otimes
            (W_{\mathrm{up}} x + b_{\mathrm{up}})
        ) + b_{\mathrm{down}}

    Args:
        channels: The number of input and output channels.
        hidden_channels: The hidden channels of the gate and up projections.
        bias: If set to ``False``, the network will not learn an additive bias.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.up_lin = Linear(
            channels, hidden_channels, bias=bias, **factory_kwargs
        )
        self.gate_lin = Linear(
            channels, hidden_channels, bias=bias, **factory_kwargs
        )
        self.down_lin = Linear(
            hidden_channels, channels, bias=bias, **factory_kwargs
        )

    def forward(self, tensor: Tensor) -> Tensor:
        r"""The forward pass.

        Args:
            tensor: The tensor.
        """
        value = self.up_lin(tensor)
        gate = F.silu(self.gate_lin(tensor))
        return self.down_lin(gate * value)
