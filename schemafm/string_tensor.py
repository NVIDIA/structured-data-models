import math
from collections.abc import Callable, Sequence
from typing import Any, cast

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
        if not data.is_contiguous():
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to be contiguous"
            )
        if offset.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to have dtype "
                f"'torch.int32' or 'torch.int64' (got '{offset.dtype}')"
            )
        if offset.dim() != 1:
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to be one-dimensional "
                f"(got {offset.dim()}D tensor)"
            )
        if not offset.is_contiguous():
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to be contiguous"
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
        if data.numel() > torch.iinfo(offset.dtype).max:
            raise ValueError(
                f"Expected 'offset' in '{cls.__name__}' to represent "
                f"{data.numel()} bytes, but '{offset.dtype}' can only "
                f"represent {torch.iinfo(offset.dtype).max} bytes"
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

    @property
    def data_offset(self) -> tuple[Tensor, Tensor]:
        if not self.is_contiguous():
            raise RuntimeError(
                f"Cannot access 'data_offset' for non-contiguous "
                f"'{self.__class__.__name__}'"
            )

        start = int(self.storage_offset())
        offset = self._offset[start : start + self.numel() + 1]
        data = self._data[offset[0] : offset[-1]]
        return data, offset - offset[0]

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
                dtype=torch.int32 if is_string else torch.int64,
            ).to(device),
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

    def to_arrow(self) -> pa.Array:
        if self.device.type != "cpu":
            raise TypeError(
                f"can't convert {self.device} device type tensor to arrow. "
                f"Use Tensor.cpu() to copy the tensor to host memory first."
            )

        data, offset = cast(StringTensor, self.contiguous()).data_offset

        return pa.Array.from_buffers(
            pa.string() if offset.dtype == torch.int32 else pa.large_string(),
            length=self.numel(),
            buffers=[
                None,
                pa.py_buffer(offset.numpy()),
                pa.py_buffer(data.numpy()),
            ],
        )

    def to_pandas(self) -> Any:
        if self.dim() != 1:
            raise ValueError(
                f"Expected '{self.__class__.__name__}' to be "
                f"one-dimensional for 'to_pandas' (got {self.dim()}D tensor)"
            )

        return self.to_arrow().to_pandas()

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

    def item(self) -> str:  # type: ignore
        if self.numel() != 1:
            raise RuntimeError(
                f"a Tensor with {self.numel()} elements cannot be converted "
                f"to a string"
            )

        start = self._offset[int(self.storage_offset())]
        end = self._offset[int(self.storage_offset()) + 1]
        return bytes(self._data[start:end].tolist()).decode("utf-8")

    def tolist(self) -> str | list[Any]:  # type: ignore
        def reshape(seq: list[str], size: tuple[int, ...]) -> str | list[Any]:
            if len(size) == 0:
                return seq[0]
            if len(size) == 1:
                return seq

            step = math.prod(size[1:])
            return [
                reshape(seq[i : i + step], size[1:])
                for i in range(0, len(seq), step)
            ]

        return reshape(self.to_arrow().to_pylist(), tuple(self.size()))

    def __str__(self) -> str:
        return self.item() if self.numel() == 1 else self.__repr__()

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return (
            f"{self.__class__.__name__}(size={tuple(self.size())}, "
            f"device='{self.device}')"
        )


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

    # Copying has two cases:
    # 1. Slice when the output layout can reuse the input storage order.
    # 2. Materialize in case of holes, overlaps, or change in memory format.
    use_slice = (
        input.numel() == 0
        or (
            memory_format == torch.preserve_format
            and torch._debug_has_internal_overlap(_layout_view(input)) == 0
        )
        or (
            memory_format == torch.contiguous_format
            and input.stride() == _contiguous_stride(input.size())
        )
    )
    if not use_slice:
        return _materialize(
            input,
            lambda x: x.clone(memory_format=memory_format),
            device=device,
            non_blocking=non_blocking,
        )

    storage_offset = int(input.storage_offset())
    span_len = _span_len(input.size(), input.stride())
    offset = input._offset[storage_offset : storage_offset + span_len + 1]
    data = input._data[offset[0] : offset[-1]].to(
        device,
        non_blocking=non_blocking,
        copy=True,
    )
    offset = (offset - offset[0]).to(device, non_blocking=non_blocking)

    return StringTensor(
        data=data,
        offset=offset,
        size=input.size(),
        stride=input.stride()
        if memory_format == torch.preserve_format
        else _contiguous_stride(input.size()),
        storage_offset=0,
    )


@implements(aten.clone.default)
def _clone(
    input: StringTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> StringTensor:
    return _to_copy(input, memory_format=memory_format)


@implements(aten.contiguous.default)
def _contiguous(
    input: StringTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> StringTensor:
    return _to_copy(input, memory_format=memory_format)


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


@implements(aten.equal.default)
def _equal(input: StringTensor, other: Tensor) -> bool:
    if not isinstance(other, StringTensor):
        return False
    if input.size() != other.size():
        return False

    data1, offset1 = cast(StringTensor, input.contiguous()).data_offset
    data2, offset2 = cast(StringTensor, other.contiguous()).data_offset

    return offset1.equal(offset2) and data1.equal(data2)


@implements(aten.allclose.default)
def _allclose(
    input: StringTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if not isinstance(other, StringTensor):
        return False
    if input.size() != other.size():
        return False

    data1, offset1 = cast(StringTensor, input.contiguous()).data_offset
    data2, offset2 = cast(StringTensor, other.contiguous()).data_offset

    return offset1.equal(offset2) and data1.allclose(
        data2, rtol=rtol, atol=atol, equal_nan=equal_nan
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


@implements(aten.unbind.int)
def _unbind(input: StringTensor, dim: int = 0) -> tuple[StringTensor, ...]:
    return tuple(
        _from_layout_view(input, view)
        for view in _layout_view(input).unbind(dim)
    )


@implements(aten.split.Tensor)
def _split(
    input: StringTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[StringTensor, ...]:
    return tuple(
        _from_layout_view(input, view)
        for view in _layout_view(input).split(split_size, dim)
    )


@implements(aten.split.sizes)
@implements(aten.split.default)
@implements(aten.split_with_sizes.default)
def _split_with_sizes(
    input: StringTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[StringTensor, ...]:
    return tuple(
        _from_layout_view(input, view)
        for view in _layout_view(input).split(tuple(split_sizes), dim)
    )


@implements(aten.expand.default)
def _expand(
    input: StringTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> StringTensor:
    view = aten.expand.default(_layout_view(input), size, implicit=implicit)
    return _from_layout_view(input, view)


@implements(aten.masked_select.default)
def _masked_select(input: StringTensor, mask: Tensor) -> StringTensor:
    return _materialize(input, lambda x: x.masked_select(mask))


@implements(aten.index_select.default)
def _index_select(
    input: StringTensor,
    dim: int,
    index: Tensor,
) -> StringTensor:
    return _materialize(input, lambda x: x.index_select(dim, index))


@implements(aten.take.default)
def _take(input: StringTensor, index: Tensor) -> StringTensor:
    return _materialize(input, lambda x: x.take(index))


@implements(aten.index.Tensor)
def _index(
    input: StringTensor,
    indices: Sequence[Tensor | None],
) -> StringTensor:
    return _materialize(input, lambda x: aten.index.Tensor(x, indices))


@implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> StringTensor:
    if len(tensors) == 0:
        raise ValueError("Expected a non-empty list of Tensors")

    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, StringTensor):
            raise TypeError(
                f"Expected '{StringTensor.__name__}' as element {i}, but got "
                f"'{type(tensor).__name__}'"
            )

    tensors = tuple(
        cast(StringTensor, tensor.contiguous()) for tensor in tensors
    )
    data_list, offsets = zip(*(tensor.data_offset for tensor in tensors))

    offset_dtype: torch.dtype = torch.int32
    if (
        any(offset.dtype == torch.int64 for offset in offsets)
        or sum(d.numel() for d in data_list) > torch.iinfo(torch.int32).max
    ):
        offset_dtype = torch.int64

    dim_size = 0
    storage_offset = 0
    start_views, end_views = [], []
    for tensor, data, offset in zip(tensors, data_list, offsets, strict=True):
        offset = offset.to(offset_dtype) + storage_offset
        dim_size += tensor.size(dim)
        storage_offset += data.numel()
        start_views.append(offset[:-1].view(tensor.size()))
        end_views.append(offset[1:].view(tensor.size()))

    size = tensors[0].size()
    dim = dim % len(size)
    size = (*size[:dim], dim_size, *size[dim + 1 :])
    offset = offsets[0].new_empty(math.prod(size) + 1, dtype=offset_dtype)
    start = torch.cat(start_views, dim=dim, out=offset[:-1].view(size))
    data = torch.cat(data_list, dim=0)

    if math.prod(tensors[0].size()[:dim]) == 1:  # Contiguous path:
        offset[-1] = storage_offset
        return StringTensor(data=data, offset=offset, size=size)

    end = torch.cat(end_views, dim=dim)
    start = torch.as_strided(start, size=(start.numel(),), stride=(1,))
    end = torch.as_strided(end, size=(end.numel(),), stride=(1,))

    offset, index = _compact(start, end)

    return StringTensor(data=data[index], offset=offset, size=size)


@implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> StringTensor:
    out = torch.cat([tensor.unsqueeze(dim) for tensor in tensors], dim=dim)
    return cast(StringTensor, out)


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


def _compact(start: Tensor, end: Tensor) -> tuple[Tensor, Tensor]:
    count = end - start

    offset = count.new_empty(count.numel() + 1)
    offset[0] = 0
    offset[1:] = count.cumsum(dim=0, dtype=count.dtype)

    local = torch.arange(  # type: ignore
        end=offset[-1],
        dtype=count.dtype,
        device=count.device,
    )
    local -= offset[:-1].repeat_interleave(
        count,
        output_size=local.numel(),
    )
    index = start.repeat_interleave(count, output_size=local.numel())
    index += local

    return offset, index


def _materialize(
    input: StringTensor,
    function: Callable[[Tensor], Tensor],
    *,
    device: torch.device | str | None = None,
    non_blocking: bool = False,
) -> StringTensor:
    # Use PyTorch's own memory-format semantics to materialize data:
    start = torch.as_strided(
        input._offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )
    start = function(start)
    assert start.storage_offset() == 0
    assert _span_len(start.size(), start.stride()) == start.numel()

    end = torch.as_strided(
        input._offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()) + 1,
    )
    end = function(end)
    assert end.storage_offset() == 0
    assert _span_len(end.size(), end.stride()) == end.numel()

    size = start.size()
    stride = start.stride()

    start = torch.as_strided(
        start,
        size=(start.numel(),),
        stride=(1,),
        storage_offset=start.storage_offset(),
    )
    end = torch.as_strided(
        end,
        size=(end.numel(),),
        stride=(1,),
        storage_offset=end.storage_offset(),
    )

    offset, index = _compact(start, end)

    return StringTensor(
        data=input._data[index].to(device, non_blocking=non_blocking),
        offset=offset.to(device, non_blocking=non_blocking),
        size=size,
        stride=stride,
        storage_offset=0,
    )
