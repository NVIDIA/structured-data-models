import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypeVar, cast

import torch
from torch import Tensor
from torch.overrides import enable_reentrant_dispatch

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(torch_function: Callable[..., Any]) -> Callable[..., Any]:

    def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


SelfVarLenTensor = TypeVar("SelfVarLenTensor", bound="VarLenTensor")


class VarLenTensor(Tensor):
    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = None

    _data: Tensor
    _offset: Tensor

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

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

    def __new__(
        cls: type[SelfVarLenTensor],
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        *,
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> SelfVarLenTensor:

        stride = stride or _contiguous_stride(size)

        if (
            cls.ALLOWED_DTYPES is not None
            and data.dtype not in cls.ALLOWED_DTYPES
        ):
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to have dtype "
                f"in '{cls.ALLOWED_DTYPES}' (got '{data.dtype}')"
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
                f"{data.numel()} elements, but '{offset.dtype}' can only "
                f"represent {torch.iinfo(offset.dtype).max} elements"
            )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=tuple(size),
            strides=tuple(stride),
            storage_offset=storage_offset,
            dtype=data.dtype,
            device=data.device,
            layout=torch.strided,
            requires_grad=False,  # Autograd lives on `_data` only.
        )

        out._data = data
        out._offset = offset

        return out

    # Properties ##############################################################

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

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data", "_offset"]
        ctx = (self.__class__, self.storage_offset())
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> "VarLenTensor":
        cls, storage_offset = ctx
        return cls(
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
            with enable_reentrant_dispatch():  # Record autograd in `_data`.
                return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for '{cls.__name__}'"
        )

    def is_shared(self) -> bool:
        return self._data.is_shared() and self._offset.is_shared()

    def share_memory_(self) -> "VarLenTensor":
        self._data.share_memory_()
        self._offset.share_memory_()
        return self

    @property
    def requires_grad(self) -> bool:
        return self._data.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._data.requires_grad_(requires_grad)

    def requires_grad_(self, mode: bool = True) -> "VarLenTensor":
        self._data.requires_grad_(mode)
        return self

    def detach_(self) -> "VarLenTensor":
        self._data.detach_()
        return self

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        return (
            f"{self.__class__.__name__}(size={tuple(self.size())}, "
            f"device='{self.device}')"
        )


@implements(aten._to_copy.default)
def _to_copy(
    input: VarLenTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,  # Ignored by PyTorch.
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:

    if memory_format is None:
        memory_format = torch.preserve_format

    if (
        dtype is not None
        and input.ALLOWED_DTYPES is not None
        and dtype not in input.ALLOWED_DTYPES
    ):
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
            dtype=dtype,
            non_blocking=non_blocking,
        )

    storage_offset = int(input.storage_offset())
    span_len = _span_len(input.size(), input.stride())
    offset = input._offset[storage_offset : storage_offset + span_len + 1]
    data = input._data[offset[0] : offset[-1]].to(
        device=device,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=True,
    )
    offset = (offset - offset[0]).to(device, non_blocking=non_blocking)

    return input.__class__(
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
    input: VarLenTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_copy(input, memory_format=memory_format)


@implements(aten.detach.default)
def _detach(input: VarLenTensor) -> VarLenTensor:
    return input.__class__(
        data=input._data.detach(),
        offset=input._offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )


@implements(aten.contiguous.default)
def _contiguous(
    input: VarLenTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> VarLenTensor:
    return _to_copy(input, memory_format=memory_format)


@implements(aten.is_pinned.default)
def _is_pinned(input: VarLenTensor) -> bool:
    return input._data.is_pinned() and input._offset.is_pinned()


@implements(aten._pin_memory.default)
def _pin_memory(input: VarLenTensor) -> VarLenTensor:
    return input.__class__(
        data=input._data.pin_memory(),
        offset=input._offset.pin_memory(),
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )


@implements(aten.equal.default)
def _equal(input: VarLenTensor, other: Tensor) -> bool:
    if input.__class__ is not other.__class__:
        return False
    if input.size() != other.size():
        return False

    data1, offset1 = cast(VarLenTensor, input.contiguous()).data_offset
    data2, offset2 = cast(VarLenTensor, other.contiguous()).data_offset

    return offset1.equal(offset2) and data1.equal(data2)


@implements(aten.allclose.default)
def _allclose(
    input: VarLenTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if input.__class__ is not other.__class__:
        return False
    if input.size() != other.size():
        return False

    data1, offset1 = cast(VarLenTensor, input.contiguous()).data_offset
    data2, offset2 = cast(VarLenTensor, other.contiguous()).data_offset

    return offset1.equal(offset2) and data1.allclose(
        data2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


@implements(aten.view.default)
def _view(input: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    view = _layout_view(input).view(tuple(size))
    return _from_layout_view(input, view)


@implements(aten._unsafe_view.default)
def _unsafe_view(input: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    view = aten._unsafe_view.default(_layout_view(input), size)
    return _from_layout_view(input, view)


@implements(aten.squeeze.default)
def _squeeze(input: VarLenTensor) -> VarLenTensor:
    view = _layout_view(input).squeeze()
    return _from_layout_view(input, view)


@implements(aten.squeeze.dim)
def _squeeze_dim(input: VarLenTensor, dim: int) -> VarLenTensor:
    view = _layout_view(input).squeeze(dim)
    return _from_layout_view(input, view)


@implements(aten.squeeze.dims)
def _squeeze_dims(input: VarLenTensor, dim: Sequence[int]) -> VarLenTensor:
    view = _layout_view(input).squeeze(tuple(dim))
    return _from_layout_view(input, view)


@implements(aten.unsqueeze.default)
def _unsqueeze(input: VarLenTensor, dim: int) -> VarLenTensor:
    view = _layout_view(input).unsqueeze(dim)
    return _from_layout_view(input, view)


@implements(aten.t.default)
def _t(input: VarLenTensor) -> VarLenTensor:
    view = _layout_view(input).t()
    return _from_layout_view(input, view)


@implements(aten.transpose.int)
def _transpose(input: VarLenTensor, dim0: int, dim1: int) -> VarLenTensor:
    view = _layout_view(input).transpose(dim0, dim1)
    return _from_layout_view(input, view)


@implements(aten.permute.default)
def _permute(input: VarLenTensor, dims: Sequence[int]) -> VarLenTensor:
    view = _layout_view(input).permute(tuple(dims))
    return _from_layout_view(input, view)


@implements(aten.select.int)
def _select(input: VarLenTensor, dim: int, index: int) -> VarLenTensor:
    view = _layout_view(input).select(dim, index)
    return _from_layout_view(input, view)


@implements(aten.slice.Tensor)
def _slice(
    input: VarLenTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> VarLenTensor:
    view = aten.slice.Tensor(_layout_view(input), dim, start, end, step)
    return _from_layout_view(input, view)


@implements(aten.narrow.default)
def _narrow(
    input: VarLenTensor,
    dim: int,
    start: int,
    length: int,
) -> VarLenTensor:
    view = _layout_view(input).narrow(dim, start, length)
    return _from_layout_view(input, view)


@implements(aten.unbind.int)
def _unbind(input: VarLenTensor, dim: int = 0) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(input, view)
        for view in _layout_view(input).unbind(dim)
    )


@implements(aten.split.Tensor)
def _split(
    input: VarLenTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(input, view)
        for view in _layout_view(input).split(split_size, dim)
    )


@implements(aten.split.sizes)
@implements(aten.split.default)
@implements(aten.split_with_sizes.default)
def _split_with_sizes(
    input: VarLenTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(input, view)
        for view in _layout_view(input).split(tuple(split_sizes), dim)
    )


@implements(aten.expand.default)
def _expand(
    input: VarLenTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> VarLenTensor:
    view = aten.expand.default(_layout_view(input), size, implicit=implicit)
    return _from_layout_view(input, view)


@implements(aten.masked_select.default)
def _masked_select(input: VarLenTensor, mask: Tensor) -> VarLenTensor:
    return _materialize(input, lambda x: x.masked_select(mask))


@implements(aten.index_select.default)
def _index_select(
    input: VarLenTensor,
    dim: int,
    index: Tensor,
) -> VarLenTensor:
    return _materialize(input, lambda x: x.index_select(dim, index))


@implements(aten.take.default)
def _take(input: VarLenTensor, index: Tensor) -> VarLenTensor:
    return _materialize(input, lambda x: x.take(index))


@implements(aten.index.Tensor)
def _index(
    input: VarLenTensor,
    indices: Sequence[Tensor | None],
) -> VarLenTensor:
    return _materialize(input, lambda x: aten.index.Tensor(x, indices))


@implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> VarLenTensor:
    if len(tensors) == 0:
        raise ValueError("Expected a non-empty list of Tensors")

    if not isinstance(tensors[0], VarLenTensor):
        raise TypeError(
            f"Expected '{VarLenTensor.__name__}' as element 0, but got "
            f"'{tensors[0].__class__.__name__}'"
        )

    tensor_cls = tensors[0].__class__
    for i, tensor in enumerate(tensors):
        if tensor.__class__ is not tensor_cls:
            raise TypeError(
                f"Expected '{tensor_cls.__name__}' as element {i}, but got "
                f"'{tensor.__class__.__name__}'"
            )

    tensors = tuple(
        cast(VarLenTensor, tensor.contiguous()) for tensor in tensors
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
    for tensor, data, offset in zip(tensors, data_list, offsets):
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
        return tensor_cls(data=data, offset=offset, size=size)

    end = torch.cat(end_views, dim=dim)
    start = torch.as_strided(start, size=(start.numel(),), stride=(1,))
    end = torch.as_strided(end, size=(end.numel(),), stride=(1,))

    offset, index = _compact(start, end)

    return tensor_cls(data=data[index], offset=offset, size=size)


@implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> VarLenTensor:
    out = torch.cat([tensor.unsqueeze(dim) for tensor in tensors], dim=dim)
    return cast(VarLenTensor, out)


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


def _layout_view(input: "VarLenTensor") -> Tensor:
    return torch.as_strided(
        input._offset,
        size=input.size(),
        stride=input.stride(),
        storage_offset=int(input.storage_offset()),
    )


def _from_layout_view(input: "VarLenTensor", view: Tensor) -> "VarLenTensor":
    return input.__class__(
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
    input: VarLenTensor,
    function: Callable[[Tensor], Tensor],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> VarLenTensor:
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

    return input.__class__(
        data=input._data[index].to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        ),
        offset=offset.to(device, non_blocking=non_blocking),
        size=size,
        stride=stride,
        storage_offset=0,
    )
