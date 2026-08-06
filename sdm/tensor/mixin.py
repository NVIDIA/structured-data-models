import abc
from collections.abc import Sequence
from typing import Self

import torch

_make_wrapper_subclass = torch.compiler.allow_in_graph(
    torch.Tensor._make_wrapper_subclass
)


def _contiguous_stride(size: Sequence[int]) -> tuple[int, ...]:
    value = 1
    stride = []
    for dim_size in reversed(size):
        stride.append(value)
        value *= torch.sym_max(dim_size, 1)
    return tuple(stride[::-1])


class DeviceMixin(abc.ABC):
    r"""Adds :class:`torch.Tensor`-like device capabilities to an object."""

    @abc.abstractmethod
    def to(self, device: torch.device | str | None) -> Self:
        r"""Perform device conversion.

        Args:
            device: The device.
        """

    def cpu(self) -> Self:
        r"""Copy data in CPU memory."""
        return self.to("cpu")

    def cuda(self, device: torch.device | str | int | None = None) -> Self:
        r"""Copy data in CUDA memory."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(torch.device("cuda", device))
        return self.to(device)

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        r"""The :class:`torch.device` where the data is."""

    @property
    def is_cpu(self) -> bool:
        r"""Whether the data is stored on the CPU."""
        return self.device.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        r"""Whether the data is stored on the GPU."""
        return self.device.type == "cuda"


def _resolve_device(
    device: torch.device | str | None,
) -> torch.device | None:
    if device is None:
        return None
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device
