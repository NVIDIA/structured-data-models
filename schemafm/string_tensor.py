import math
from collections.abc import Callable, Sequence
from typing import Any

import pyarrow as pa
import torch
from torch import Tensor

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def _contiguous_stride(size: Sequence[int]) -> tuple[int, ...]:
    value = 1
    stride = []
    for dim_size in reversed(size):
        stride.append(value)
        value *= dim_size
    return tuple(stride[::-1])


def _max_index(
    size: Sequence[int],
    stride: Sequence[int],
    storage_offset: int,
) -> int:
    if math.prod(size) == 0:
        return storage_offset
    return storage_offset + sum(
        (dim_size - 1) * dim_stride
        for dim_size, dim_stride in zip(size, stride, strict=True)
    )


def implements(torch_function: Callable[..., Any]) -> Callable[..., Any]:

    def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


class StringTensor(Tensor):
    _data: Tensor
    _offset: Tensor

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    def __init__(
        cls,
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        *,
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> None:
        pass

    @staticmethod
    def __new__(
        cls,
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        *,
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> "StringTensor":

        stride = stride or _contiguous_stride(size)

        if data.dtype != torch.uint8:
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to have dtype "
                f"'torch.uint8' (got '{data.dtype}')"
            )
        if data.dim() != 1:
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to be one-dimensional "
                f"(got {data.dim()}D tensor)"
            )
        if offset.dtype != torch.long:  # TODO Relax 8-byte offset restriction.
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to have dtype "
                f"'torch.int64' (got '{offset.dtype}')"
            )
        if offset.dim() != 1:
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to be one-dimensional "
                f"(got {offset.dim()}D tensor)"
            )
        if data.device != offset.device:
            raise ValueError(
                f"Expected 'data' and 'offset' in '{cls.__name__}' to be on "
                f"the same device (got '{data.device}' and '{offset.device}')"
            )
        if len(size) != len(stride):
            raise ValueError(
                f"Expected 'size' and 'stride' in '{cls.__name__}' to have "
                f"the same length (got {len(size)} and {len(stride)})"
            )
        if storage_offset < 0 or storage_offset >= offset.numel():
            raise ValueError(
                f"'storage_offset' in '{cls.__name__}' is out of bounds (got "
                f"{storage_offset}, but expected [0, {offset.numel() - 1}])"
            )
        if math.prod(size) > 0:
            max_index = _max_index(size, stride, storage_offset)
            if max_index >= offset.numel() - 1:
                raise ValueError(
                    f"'offset' in '{cls.__name__}' is out of bounds (got "
                    f"{offset.numel() - 1} elements, but expected at least "
                    f"{max_index + 1} elements)"
                )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=tuple(size),
            strides=tuple(stride),
            storage_offset=storage_offset,
            dtype=torch.uint8,
            device=data.device,
            layout=torch.strided,
            requires_grad=False,
        )

        out._data = data
        out._offset = offset

        return out

    @classmethod
    def from_arrow(
        cls,
        data: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> "StringTensor":
        if isinstance(data, pa.ChunkedArray):
            if data.num_chunks == 1:
                data = data.chunk(0)
            else:
                data = data.combine_chunks()

        if not isinstance(data, pa.Array):
            raise TypeError(
                f"Expected 'data' in '{cls.__name__}.from_arrow' to be a "
                f"'pyarrow.Array' or 'pyarrow.ChunkedArray' "
                f"(got '{type(data).__name__}')"
            )

        is_string = pa.types.is_string(data.type)
        is_large_string = pa.types.is_large_string(data.type)
        if not is_string and not is_large_string:
            raise TypeError(
                f"Expected 'data' in '{cls.__name__}.from_arrow' to have "
                f"'string' or 'large_string' type (got '{data.type}')"
            )

        if size is None:
            size = (len(data),)
        elif math.prod(size) != len(data):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_arrow' to contain "
                f"{len(data)} elements (got {math.prod(size)})"
            )

        buffers = data.buffers()

        return cls(
            data=torch.frombuffer(buffers[2], dtype=torch.uint8).to(device)
            if buffers[2].size > 0
            else torch.empty(0, dtype=torch.uint8, device=device),
            offset=torch.frombuffer(
                buffer=buffers[1],
                dtype=torch.int if is_string else torch.long,
            ).to(device, torch.long),
            size=size,
            storage_offset=data.offset,
        )

    @classmethod
    def from_strings(
        cls,
        data: str | Sequence[Any],
        *,
        device: torch.device | str | None = None,
    ) -> "StringTensor":
        def flatten(data: Any) -> tuple[int, ...]:
            if isinstance(data, str):
                return ()
            if not isinstance(data, Sequence):
                raise TypeError(f"'{cls.__name__}' data must contain strings")
            if len(data) == 0:
                return (0,)

            if not isinstance(data[0], Sequence) or isinstance(data[0], str):
                values.extend(data)
                return (len(data),)

            child_size: tuple[int, ...] | None = None
            for item in data:
                item_size = flatten(item)
                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"'{cls.__name__}' data must be rectangular"
                    )

            assert child_size is not None
            return (len(data), *child_size)

        if isinstance(data, str):
            values: list[str] = [data]
            size: tuple[int, ...] = ()
        else:
            values = []
            size = flatten(data)

        return cls.from_arrow(
            data=pa.array(values, type=pa.large_string()),
            device=device,
            size=size,
        )

    @classmethod
    def from_pandas(
        cls,
        data: Any,
        *,
        device: torch.device | str | None = None,
    ) -> "StringTensor":
        import pandas as pd

        if not isinstance(data, pd.Series):
            raise TypeError(
                f"Expected 'data' in '{cls.__name__}.from_pandas' to be a "
                f"'pandas.Series' (got '{type(data).__name__}')"
            )

        return cls.from_arrow(
            data=data.astype("string[pyarrow]").array.__arrow_array__(),
            device=device,
        )

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data", "_offset"]
        ctx = (self.storage_offset(),)
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> "StringTensor":
        (storage_offset,) = ctx
        return StringTensor(
            data=inner_tensors["_data"],
            offset=inner_tensors["_offset"],
            size=outer_size,
            stride=outer_stride,
            storage_offset=storage_offset,
        )

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for '{cls.__name__}'"
        )

    def is_shared(self) -> bool:
        return self._data.is_shared() and self._offset.is_shared()

    def share_memory_(self) -> "StringTensor":
        self._data.share_memory_()
        self._offset.share_memory_()
        return self


@implements(aten.clone.default)
def _clone(
    input: StringTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> StringTensor:
    return _to_copy(input, memory_format=memory_format)


@implements(aten._to_copy.default)
def _to_copy(
    input: StringTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> StringTensor:

    if memory_format is None:
        memory_format = torch.preserve_format

    if dtype is not None and dtype != torch.uint8:
        raise TypeError(
            f"Cannot convert '{input.__class__.__name__}' to dtype '{dtype}'"
        )
    if layout is not None and layout != torch.strided:
        raise TypeError(
            f"Cannot convert '{input.__class__.__name__}' to layout '{layout}'"
        )
    if memory_format not in (torch.preserve_format, torch.contiguous_format):
        raise ValueError(
            f"Unsupported memory format '{memory_format}' for "
            f"'{input.__class__.__name__}.clone'"
        )

    if input.stride() != _contiguous_stride(input.size()):
        raise NotImplementedError(  # TODO
            f"Cannot copy non-contiguous '{input.__class__.__name__}'"
        )

    storage_offset = int(input.storage_offset())
    if input.numel() == 0:
        data = input._data.new_empty(0)
        offset = input._offset.new_zeros(1)
    else:
        max_index = _max_index(input.size(), input.stride(), storage_offset)
        offset = input._offset[storage_offset : max_index + 2]
        byte_start = int(offset[0])
        byte_end = int(offset[-1])
        data = input._data[byte_start:byte_end]
        offset = offset - offset[0]

    if device is not None and device != data.device:
        data = data.to(device, non_blocking=non_blocking)
        offset = offset.to(device, non_blocking=non_blocking)
    elif input.numel() > 0:
        data = data.clone()

    return StringTensor(
        data=data,
        offset=offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=0,
    )


@implements(aten.is_pinned.default)
def _is_pinned(
    input: StringTensor,
    device: torch.device | str | None = None,
) -> bool:
    return input._data.is_pinned(device=device) and input._offset.is_pinned(
        device=device
    )


@implements(aten._pin_memory.default)
def _pin_memory(
    input: StringTensor,
    device: torch.device | str | None = None,
) -> StringTensor:
    if device is None:
        data = input._data.pin_memory()
        offset = input._offset.pin_memory()
    else:
        data = input._data.pin_memory(device=device)
        offset = input._offset.pin_memory(device=device)

    return StringTensor(
        data=data,
        offset=offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )
