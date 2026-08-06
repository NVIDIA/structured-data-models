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


def _storage_delta_coordinates(
    size: Sequence[int],
    stride: Sequence[int],
    storage_delta: int | torch.SymInt,
    *,
    component: str,
) -> tuple[int | torch.SymInt, ...]:
    coordinates: list[int | torch.SymInt] = [0] * len(size)
    remaining = storage_delta
    dims = sorted(
        range(len(size)),
        key=lambda dim: stride[dim],
        reverse=True,
    )
    for dim in dims:
        if size[dim] <= 1 or stride[dim] == 0:
            continue
        coordinate = remaining // stride[dim]
        if coordinate < 0 or coordinate >= size[dim]:
            raise RuntimeError("Output layout points outside the input view")
        coordinates[dim] = coordinate
        remaining -= coordinate * stride[dim]
    if remaining != 0:
        raise RuntimeError(
            f"Output layout cannot be replayed on the {component}"
        )
    return tuple(coordinates)


def _replay_as_strided(
    target: torch.Tensor,
    *,
    source_size: Sequence[int],
    source_stride: Sequence[int],
    size: Sequence[int],
    stride: Sequence[int],
    storage_delta: int | torch.SymInt,
    component: str,
    target_storage_offset: int | torch.SymInt | None = None,
) -> torch.Tensor:
    source_size = tuple(source_size)
    source_stride = tuple(source_stride)
    size = tuple(size)
    stride = tuple(stride)
    suffix_size = target.size()[len(source_size) :]
    suffix_stride = target.stride()[len(source_size) :]

    logical = torch.as_strided(
        torch.empty(0, dtype=torch.uint8, device="meta"),
        size=source_size,
        stride=source_stride,
    )
    if storage_delta == 0:
        try:
            view = logical.view(size)
        except RuntimeError:
            pass
        else:
            if view.stride() == stride:
                try:
                    target_view = target.view((*size, *suffix_size))
                except RuntimeError:
                    pass
                else:
                    return target_view

    coordinates = _storage_delta_coordinates(
        source_size,
        source_stride,
        storage_delta,
        component=component,
    )
    if target_storage_offset is None:
        target_storage_offset = target.storage_offset()
    output_storage_offset = target_storage_offset + sum(
        coordinate * dim_stride
        for coordinate, dim_stride in zip(coordinates, target.stride())
    )

    output_stride = []
    mapped_steps: list[tuple[int, int | torch.SymInt, int]] = []
    for dim_size, dim_stride in zip(size, stride):
        if dim_stride == 0:
            output_stride.append(0)
            continue

        fallback: tuple[int, int | torch.SymInt] | None = None
        for source_dim, input_stride in enumerate(source_stride):
            if input_stride <= 0 or dim_stride % input_stride != 0:
                continue
            step = dim_stride // input_stride
            if fallback is None:
                fallback = source_dim, step
            if (
                dim_size <= 1
                or coordinates[source_dim] + (dim_size - 1) * step
                < source_size[source_dim]
            ):
                break
        else:
            if fallback is None:
                if dim_size <= 1:
                    output_stride.append(0)
                    continue
                raise RuntimeError(
                    f"Output layout cannot be replayed on the {component}"
                )
            source_dim, step = fallback
        output_stride.append(target.stride(source_dim) * step)
        mapped_steps.append((source_dim, step, dim_size))

    distances = tuple(
        sum(
            (dim_size - 1) * step
            for mapped_dim, step, dim_size in mapped_steps
            if mapped_dim == source_dim
        )
        for source_dim in range(len(source_size))
    )
    source_dims = sorted(
        (
            dim
            for dim in range(len(source_size))
            if source_size[dim] > 1 and source_stride[dim] > 0
        ),
        key=lambda dim: source_stride[dim],
    )
    if source_dims:
        reference = source_dims[0]
        group_extent = (
            coordinates[reference] + distances[reference]
        ) * source_stride[reference]
        for source_dim in source_dims[1:]:
            input_stride = source_stride[source_dim]
            if group_extent < input_stride:
                reference = source_dim
                group_extent = (
                    coordinates[source_dim] + distances[source_dim]
                ) * input_stride
                continue
            if (
                target.stride(source_dim) * source_stride[reference]
                != target.stride(reference) * input_stride
            ):
                raise RuntimeError(
                    f"Output layout cannot be replayed on the {component}"
                )
            group_extent += (
                coordinates[source_dim] + distances[source_dim]
            ) * input_stride

    return torch.ops.aten.as_strided.default(
        target,
        (*size, *suffix_size),
        (*output_stride, *suffix_stride),
        output_storage_offset,
    )


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
