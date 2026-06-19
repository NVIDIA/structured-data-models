import math
from collections.abc import Callable, Sequence
from typing import Any

import pyarrow as pa
import torch
from torch import Tensor

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


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
        if storage_offset < 0:
            raise ValueError(
                f"Expected 'storage_offset' in '{cls.__name__}' to be "
                f"non-negative"
            )
        if storage_offset + _span_len(size, stride) >= offset.numel():
            raise ValueError(
                f"'offset' in '{cls.__name__}' is out of bounds (got "
                f"{offset.numel()} entries, but expected at least "
                f"{storage_offset + _span_len(size, stride) + 1} entries)"
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
    pin_memory: bool = False,  # Ignored by PyTorch.
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

    # `StringTensor` stores one physical 1D blob in `_offset`/`_data`.
    # Logical tensor positions are mapped into that blob via
    # `size`/`stride`/`storage_offset`, just like regular strided tensors.
    # Copying therefore has two cases:
    # 1. Slice one physical span when the requested output layout can reuse the
    #    input storage order.
    # 2. Gather logical elements in physical order when holes, overlaps, or
    #    `contiguous_format` require materialization.
    storage_offset = int(input.storage_offset())
    start = torch.as_strided(
        input._offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=storage_offset,
    )
    use_slice = (
        input.numel() == 0
        or (
            memory_format == torch.preserve_format
            and torch._debug_has_internal_overlap(start) == 0
        )
        or (
            memory_format == torch.contiguous_format
            and input.stride() == _contiguous_stride(input.size())
        )
    )
    if use_slice:
        span_len = _span_len(input.size(), input.stride())
        offset = input._offset[storage_offset : storage_offset + span_len + 1]
        data = input._data[offset[0] : offset[-1]].to(
            device,
            non_blocking=non_blocking,
            copy=True,
        )
        offset = (offset - offset[0]).to(device, non_blocking=non_blocking)
        if memory_format == torch.preserve_format:
            stride = input.stride()
        else:
            stride = _contiguous_stride(input.size())
    else:
        # Use PyTorch's own memory-format semantics to materialize data:
        start = start.clone(memory_format=memory_format)
        stride = start.stride()
        start = torch.as_strided(  # Physical 1D representation.
            start,
            size=(start.numel(),),
            stride=(1,),
            storage_offset=start.storage_offset(),
        )
        end = torch.as_strided(
            input._offset,
            size=input.size(),
            stride=input.stride(),
            storage_offset=storage_offset + 1,
        ).clone(memory_format=memory_format)
        end = torch.as_strided(  # Physical 1D representation.
            end,
            size=(end.numel(),),
            stride=(1,),
            storage_offset=end.storage_offset(),
        )
        count = end - start

        offset = count.new_empty(count.numel() + 1)
        offset[0] = 0
        offset[1:] = count.cumsum(dim=0)

        local = torch.arange(offset[-1], device=count.device)  # type: ignore
        local -= offset[:-1].repeat_interleave(
            count, output_size=local.numel()
        )
        index = start.repeat_interleave(count, output_size=local.numel())
        index += local

        data = input._data[index].to(device, non_blocking=non_blocking)
        offset = offset.to(device, non_blocking=non_blocking)

    return StringTensor(
        data=data,
        offset=offset,
        size=input.size(),
        stride=stride,
        storage_offset=0,
    )


@implements(aten.is_pinned.default)
def _is_pinned(input: StringTensor) -> bool:
    return input._data.is_pinned() and input._offset.is_pinned()


@implements(aten._pin_memory.default)
def _pin_memory(input: StringTensor) -> StringTensor:
    return StringTensor(
        data=input._data.pin_memory(),
        offset=input._offset.pin_memory(),
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )


@implements(aten.view.default)
def _view(input: StringTensor, size: Sequence[int]) -> StringTensor:
    view = _layout_view(input).view(tuple(size))
    return _from_layout_view(input, view)


@implements(aten._unsafe_view.default)
def _unsafe_view(input: StringTensor, size: Sequence[int]) -> StringTensor:
    view = aten._unsafe_view.default(_layout_view(input), size)
    return _from_layout_view(input, view)


@implements(aten.squeeze.default)
def _squeeze(input: StringTensor) -> StringTensor:
    view = _layout_view(input).squeeze()
    return _from_layout_view(input, view)


@implements(aten.squeeze.dim)
def _squeeze_dim(input: StringTensor, dim: int) -> StringTensor:
    view = _layout_view(input).squeeze(dim)
    return _from_layout_view(input, view)


@implements(aten.squeeze.dims)
def _squeeze_dims(input: StringTensor, dim: Sequence[int]) -> StringTensor:
    view = _layout_view(input).squeeze(tuple(dim))
    return _from_layout_view(input, view)


@implements(aten.unsqueeze.default)
def _unsqueeze(input: StringTensor, dim: int) -> StringTensor:
    view = _layout_view(input).unsqueeze(dim)
    return _from_layout_view(input, view)


@implements(aten.t.default)
def _t(input: StringTensor) -> StringTensor:
    view = _layout_view(input).t()
    return _from_layout_view(input, view)


@implements(aten.transpose.int)
def _transpose(input: StringTensor, dim0: int, dim1: int) -> StringTensor:
    view = _layout_view(input).transpose(dim0, dim1)
    return _from_layout_view(input, view)


@implements(aten.permute.default)
def _permute(input: StringTensor, dims: Sequence[int]) -> StringTensor:
    view = _layout_view(input).permute(tuple(dims))
    return _from_layout_view(input, view)


@implements(aten.select.int)
def _select(input: StringTensor, dim: int, index: int) -> StringTensor:
    view = _layout_view(input).select(dim, index)
    return _from_layout_view(input, view)


@implements(aten.slice.Tensor)
def _slice(
    input: StringTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> StringTensor:
    view = aten.slice.Tensor(_layout_view(input), dim, start, end, step)
    return _from_layout_view(input, view)


@implements(aten.narrow.default)
def _narrow(
    input: StringTensor,
    dim: int,
    start: int,
    length: int,
) -> StringTensor:
    view = _layout_view(input).narrow(dim, start, length)
    return _from_layout_view(input, view)


@implements(aten.expand.default)
def _expand(
    input: StringTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> StringTensor:
    view = aten.expand.default(_layout_view(input), size, implicit=implicit)
    return _from_layout_view(input, view)


# Helpers #####################################################################


def _contiguous_stride(size: Sequence[int]) -> tuple[int, ...]:
    value = 1
    stride = []
    for dim_size in reversed(size):
        stride.append(value)
        value *= dim_size
    return tuple(stride[::-1])


def _span_len(size: Sequence[int], stride: Sequence[int]) -> int:
    if math.prod(size) == 0:
        return 0
    return 1 + sum(
        (dim_size - 1) * dim_stride
        for dim_size, dim_stride in zip(size, stride)
    )


def _layout_view(input: "StringTensor") -> Tensor:
    return torch.as_strided(
        input._offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )


def _from_layout_view(input: "StringTensor", view: Tensor) -> "StringTensor":
    return StringTensor(
        data=input._data,
        offset=input._offset,
        size=view.size(),
        stride=view.stride(),
        storage_offset=int(view.storage_offset()),
    )
