import abc

import torch
from typing_extensions import Self


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
