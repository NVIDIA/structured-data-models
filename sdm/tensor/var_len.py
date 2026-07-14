from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, SupportsIndex, cast

import pyarrow as pa
import torch
from torch import Tensor
from torch.overrides import enable_reentrant_dispatch
from typing_extensions import Self, override

from sdm.tensor.io import ARROW_TORCH_DTYPES, to_arrow

aten = torch.ops.aten


def preserve_view_inference_mode(fn: Callable) -> Callable:
    r"""Preserve input inference state for tensor view operations."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(args[0].is_inference()):
            return fn(*args, **kwargs)

    return wrapper


class VarLenTensor(Tensor):
    r"""A :class:`torch.Tensor` for rectangular variable-length values.

    Values are stored in a flat contiguous ``data`` tensor and indexed by an
    ``offset`` tensor.

    .. code-block:: python

        import torch
        from sdm import VarLenTensor

        tensor = VarLenTensor(
            data=torch.tensor([1, 2, 3, 4, 5, 6]),
            offset=torch.tensor([0, 2, 5, 5, 6]),
            size=(2, 2),
        )

    Args:
        data: Flat contiguous tensor containing all element values.
        offset: One-dimensional offsets into ``data``.
        size: The shape of the tensor.
        stride: The stride of the tensor.
        storage_offset: The offset into the logical ``offset`` storage.
    """

    ALLOWED_DTYPES: ClassVar[tuple[torch.dtype, ...] | None] = None
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _data: Tensor
    _offset: Tensor

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        self,
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> None:
        pass

    def __new__(
        cls,
        data: Tensor,
        offset: Tensor,
        size: Sequence[int],
        stride: Sequence[int] | None = None,
        storage_offset: int = 0,
    ) -> Self:
        r"""Create a tensor wrapper."""
        size = tuple(size)
        if any(dim_size < -1 for dim_size in size):
            raise ValueError(f"Invalid shape dimensions (got '{size}')")
        if size.count(-1) > 1:
            raise ValueError("Only one dimension can be inferred")

        if -1 in size:
            if stride is not None:
                raise ValueError("Can't infer size when stride is given")

            numel = offset.numel() - 1 - storage_offset
            known = math.prod(dim_size for dim_size in size if dim_size != -1)
            if numel < 0 or known == 0 or numel % known != 0:
                raise ValueError(
                    f"Shape '{size}' is invalid for input of size {numel}"
                )

            dim = size.index(-1)
            size = (*size[:dim], numel // known, *size[dim + 1 :])

        stride = _contiguous_stride(size) if stride is None else tuple(stride)
        if any(dim_stride < 0 for dim_stride in stride):
            raise ValueError(
                f"Negative strides are not supported (got '{stride}')"
            )

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

        # NOTE We do not validate offset values here (e.g., monotonicity) due
        # to device synchronization.

        out = Tensor._make_wrapper_subclass(
            cls,
            size=size,
            strides=stride,
            storage_offset=storage_offset,
            dtype=data.dtype,
            device=data.device,
            requires_grad=False,  # Autograd lives on `_data` only.
        )

        out._data = data
        out._offset = offset

        return out

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        *,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Wrap a dense tensor as fixed-size variable-length elements.

        Args:
            tensor: The dense tensor.
            offset_dtype: The dtype of the ``offset`` tensor.
        """
        data = tensor
        if tensor.stride() != (1,) or int(tensor.storage_offset()) != 0:
            span_len = _span_len(tensor.size(), tensor.stride())
            data = torch.as_strided(
                tensor,
                size=(int(data.storage_offset()) + span_len,),
                stride=(1,),
                storage_offset=0,
            )
        offset = torch.arange(
            data.numel() + 1,
            dtype=offset_dtype,
            device=data.device,
        )
        return cls(
            data=data,
            offset=offset,
            size=tensor.size(),
            stride=tensor.stride(),
            storage_offset=int(tensor.storage_offset()),
        )

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a list :class:`pyarrow.Array`.

        .. code-block:: python

            import pyarrow as pa
            from sdm import VarLenTensor

            array = pa.array([[1, 2], [3, 4, 5], [], [6]])
            tensor = VarLenTensor.from_arrow(array)

        Args:
            array: The list :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            size: The shape of the tensor.
            device: The device.
        """
        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

        if size is None:
            size = (len(array),)
        elif math.prod(size) != len(array):
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}.from_arrow' to contain "
                f"{len(array)} elements (got {math.prod(size)})"
            )

        is_list = pa.types.is_list(array.type)
        is_large_list = pa.types.is_large_list(array.type)
        if not is_list and not is_large_list:
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to have "
                f"'list' or 'large_list' type (got '{array.type}')"
            )

        if array.null_count > 0 or array.values.null_count > 0:
            raise ValueError(f"'{cls.__name__}' cannot represent null values")

        dtype = ARROW_TORCH_DTYPES.get(array.values.type)
        if dtype is None:
            raise TypeError(f"Unsupported value type '{array.values.type}'")
        offset_dtype = torch.int32 if is_list else torch.int64

        buffer = array.values.buffers()[1]
        if buffer is not None and buffer.size > 0:
            data = torch.frombuffer(buffer, dtype=dtype)
            start = array.values.offset
            data = data[start : start + len(array.values)].to(device)
        else:
            data = torch.empty(0, dtype=dtype, device=device)

        offset = torch.frombuffer(array.buffers()[1], dtype=offset_dtype)
        offset = offset.to(device)

        return cls(
            data=data,
            offset=offset,
            size=size,
            storage_offset=array.offset,
        )

    def to_arrow(self) -> pa.Array:
        r"""Convert this tensor to a flat :class:`pyarrow.Array`."""
        tensor = cast(VarLenTensor, self.detach().contiguous().cpu())
        array = to_arrow(tensor._data)

        return pa.Array.from_buffers(
            pa.list_(array.type)
            if tensor._offset.dtype == torch.int32
            else pa.large_list(array.type),
            length=tensor.numel(),
            buffers=[None, pa.py_buffer(tensor._offset.numpy())],
            children=[array],
            offset=int(tensor.storage_offset()),
        )

    @classmethod
    def from_list(
        cls,
        values: Sequence[Any],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        offset_dtype: torch.dtype = torch.int64,
    ) -> Self:
        r"""Create tensor from a rectangular Python list.

        .. code-block:: python

            from sdm import VarLenTensor

            tensor = VarLenTensor.from_list([
                [[1, 2], [3, 4, 5]],
                [[], [6]],
            ])

        Args:
            values: The rectangular Python list.
            dtype: The dtype of the ``value`` tensor.
            device: The device.
            offset_dtype: The dtype of the ``offset`` tensor.
        """

        def is_sequence(value: Any) -> bool:
            return isinstance(value, Sequence) and not isinstance(
                value, str | bytes | bytearray
            )

        def flatten(seq: Sequence[Any]) -> tuple[int, ...]:
            if len(seq) == 0:
                offset.append(len(data))
                return ()

            if not is_sequence(seq[0]):
                data.extend(seq)
                offset.append(len(data))
                return ()

            child_size: tuple[int, ...] | None = None
            for item in seq:
                if not is_sequence(item):
                    raise ValueError(
                        f"'{cls.__name__}' data must be rectangular"
                    )
                item_size = flatten(cast(Sequence[Any], item))
                if child_size is None:
                    child_size = item_size
                elif item_size != child_size:
                    raise ValueError(
                        f"'{cls.__name__}' data must be rectangular"
                    )

            assert child_size is not None
            return (len(seq), *child_size)

        data: list[Any] = []
        offset = [0]
        size = (0,) if len(values) == 0 else flatten(values)

        return cls(
            data=torch.tensor(data, dtype=dtype, device=device),
            offset=torch.tensor(offset, dtype=offset_dtype, device=device),
            size=size,
        )

    # Properties ##############################################################

    @property
    def data_offset(self) -> tuple[Tensor, Tensor]:
        r"""Return contiguous data and normalized offsets.

        Returns:
            ``(data, offset)`` tuple.
        """
        if not self.is_contiguous():
            raise RuntimeError(
                f"Can't access 'data_offset' for non-contiguous "
                f"'{self.__class__.__name__}'"
            )

        start = int(self.storage_offset())
        offset = self._offset[start : start + self.numel() + 1]
        data = self._data[offset[0] : offset[-1]]
        if offset.is_cpu and int(offset[0]) == 0:
            return data, offset
        return data, offset - offset[0]

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        r"""Register a ``__torch_dispatch__`` implementation.

        See PyTorch's
        :ref:`calling convention <torch-dispatch-calling-convention>`.
        """
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

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
    ) -> VarLenTensor:
        cls, storage_offset = ctx
        return cls(
            data=inner_tensors["_data"],
            offset=inner_tensors["_offset"],
            size=outer_size,
            stride=outer_stride,
            storage_offset=storage_offset,
        )

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (
            self._data,
            self._offset,
            tuple(self.size()),
            tuple(self.stride()),
            int(self.storage_offset()),
        )
        return (self.__class__, args)

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if (handler := cls.HANDLED_FUNCTIONS.get(func)) is not None:
            with enable_reentrant_dispatch():  # Record autograd in `_data`.
                return handler(*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for '{cls.__name__}'"
        )

    @override
    def is_shared(self) -> bool:
        return self._data.is_shared() and self._offset.is_shared()

    @override
    def share_memory_(self) -> Self:
        self._data.share_memory_()
        self._offset.share_memory_()
        return self

    @property
    @override
    def grad(self) -> Tensor | None:
        return self._data.grad

    @property
    @override
    def requires_grad(self) -> bool:
        return self._data.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool) -> None:
        self._data.requires_grad_(requires_grad)

    @override
    def requires_grad_(self, mode: bool = True) -> Self:
        self._data.requires_grad_(mode)
        return self

    @override
    def detach_(self) -> Self:
        raise RuntimeError(
            f"Can't detach a '{self.__class__.__name__} in-place. Use "
            f"'detach() instead."
        )

    @override
    def tolist(self) -> Any:
        def reshape(values: list[Any], size: tuple[int, ...]) -> Any:
            if len(size) == 0:
                return values[0]
            if len(size) == 1:
                return values

            step = math.prod(size[1:])
            return [
                reshape(values[i * step : (i + 1) * step], size[1:])
                for i in range(size[0])
            ]

        tensor = cast(VarLenTensor, self.detach().cpu())
        values = tensor.to_arrow().to_pylist()
        return reshape(values, tuple(self.size()))

    @override
    def item(self) -> list[Any]:  # type: ignore
        if self.numel() != 1:
            raise RuntimeError(
                f"'{self.__class__.__name__}' with {self.numel()} "
                f"elements cannot be converted to a single item"
            )
        return self.view(-1).tolist()[0]

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        # TODO Support tensor content printing.
        out = f"{self.__class__.__name__}(..."
        out += f", size={tuple(self.size())}"
        out += f", dtype={self.dtype}"
        if not self.is_cpu:
            out += f", device={self.device}"
        if self._data.grad_fn is not None:
            out += f", grad_fn=<{type(self._data.grad_fn).__name__}>"
        elif self.requires_grad:
            out += ", requires_grad=True"
        out += ")"
        return out


@VarLenTensor.implements(aten.alias.default)
@preserve_view_inference_mode
def _alias(inp: VarLenTensor) -> VarLenTensor:
    return inp.__class__(
        data=aten.alias.default(inp._data),
        offset=aten.alias.default(inp._offset),
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


@VarLenTensor.implements(aten._to_copy.default)
def _to_copy(
    inp: VarLenTensor,
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
        and inp.ALLOWED_DTYPES is not None
        and dtype not in inp.ALLOWED_DTYPES
    ):
        raise TypeError(
            f"Can't convert '{inp.__class__.__name__}' to dtype '{dtype}'"
        )
    if layout is not None and layout != torch.strided:
        raise TypeError(
            f"Can't convert '{inp.__class__.__name__}' to layout '{layout}'"
        )
    if memory_format not in (torch.preserve_format, torch.contiguous_format):
        raise ValueError(
            f"Unsupported memory format '{memory_format}' for "
            f"'{inp.__class__.__name__}.clone'"
        )

    # Copying has two cases:
    # 1. Slice when the output layout can reuse the input storage order.
    # 2. Materialize in case of holes, overlaps, or change in memory format.
    use_slice = (
        inp.numel() == 0
        or (
            memory_format == torch.preserve_format
            and torch._debug_has_internal_overlap(_layout_view(inp)) == 0
        )
        or (
            memory_format == torch.contiguous_format
            and inp.stride() == _contiguous_stride(inp.size())
        )
    )
    if not use_slice:
        return _materialize(
            inp,
            lambda x: x.clone(memory_format=memory_format),
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )

    storage_offset = int(inp.storage_offset())
    span_len = _span_len(inp.size(), inp.stride())
    offset = inp._offset[storage_offset : storage_offset + span_len + 1]
    data = inp._data[offset[0] : offset[-1]].to(
        device=device,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=True,
    )
    offset = (offset - offset[0]).to(device, non_blocking=non_blocking)

    return inp.__class__(
        data=data,
        offset=offset,
        size=inp.size(),
        stride=inp.stride()
        if memory_format == torch.preserve_format
        else _contiguous_stride(inp.size()),
        storage_offset=0,
    )


@VarLenTensor.implements(aten.to.dtype_layout)
def _to_dtype_layout(
    inp: VarLenTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_copy(
        inp,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=bool(pin_memory),
        non_blocking=non_blocking,
        memory_format=memory_format,
    )


@VarLenTensor.implements(aten.clone.default)
def _clone(
    inp: VarLenTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> VarLenTensor:
    return _to_copy(inp, memory_format=memory_format)


@VarLenTensor.implements(aten.detach.default)
@preserve_view_inference_mode
def _detach(inp: VarLenTensor) -> VarLenTensor:
    return inp.__class__(
        data=inp._data.detach(),
        offset=inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


@VarLenTensor.implements(aten.contiguous.default)
def _contiguous(
    inp: VarLenTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> VarLenTensor:
    return _to_copy(inp, memory_format=memory_format)


@VarLenTensor.implements(aten.is_pinned.default)
def _is_pinned(inp: VarLenTensor) -> bool:
    return inp._data.is_pinned() and inp._offset.is_pinned()


@VarLenTensor.implements(aten._pin_memory.default)
def _pin_memory(inp: VarLenTensor) -> VarLenTensor:
    return inp.__class__(
        data=inp._data.pin_memory(),
        offset=inp._offset.pin_memory(),
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


@VarLenTensor.implements(aten.equal.default)
def _equal(inp: VarLenTensor, other: Tensor) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    data1, offset1 = cast(VarLenTensor, inp.contiguous()).data_offset
    data2, offset2 = cast(VarLenTensor, other.contiguous()).data_offset

    return offset1.equal(offset2) and data1.equal(data2)


@VarLenTensor.implements(aten.allclose.default)
def _allclose(
    inp: VarLenTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    data1, offset1 = cast(VarLenTensor, inp.contiguous()).data_offset
    data2, offset2 = cast(VarLenTensor, other.contiguous()).data_offset

    return offset1.equal(offset2) and data1.allclose(
        data2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


@VarLenTensor.implements(aten.view.default)
@preserve_view_inference_mode
def _view(inp: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    view = _layout_view(inp).view(tuple(size))
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten._unsafe_view.default)
@preserve_view_inference_mode
def _unsafe_view(inp: VarLenTensor, size: Sequence[int]) -> VarLenTensor:
    view = aten._unsafe_view.default(_layout_view(inp), size)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.squeeze.default)
@preserve_view_inference_mode
def _squeeze(inp: VarLenTensor) -> VarLenTensor:
    view = _layout_view(inp).squeeze()
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.squeeze.dim)
@preserve_view_inference_mode
def _squeeze_dim(inp: VarLenTensor, dim: int) -> VarLenTensor:
    view = _layout_view(inp).squeeze(dim)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.squeeze.dims)
@preserve_view_inference_mode
def _squeeze_dims(inp: VarLenTensor, dim: Sequence[int]) -> VarLenTensor:
    view = _layout_view(inp).squeeze(tuple(dim))
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.unsqueeze.default)
@preserve_view_inference_mode
def _unsqueeze(inp: VarLenTensor, dim: int) -> VarLenTensor:
    view = _layout_view(inp).unsqueeze(dim)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.expand.default)
@preserve_view_inference_mode
def _expand(
    inp: VarLenTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> VarLenTensor:
    view = aten.expand.default(_layout_view(inp), size, implicit=implicit)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.t.default)
@preserve_view_inference_mode
def _t(inp: VarLenTensor) -> VarLenTensor:
    view = _layout_view(inp).t()
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.transpose.int)
@preserve_view_inference_mode
def _transpose(inp: VarLenTensor, dim0: int, dim1: int) -> VarLenTensor:
    view = _layout_view(inp).transpose(dim0, dim1)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.permute.default)
@preserve_view_inference_mode
def _permute(inp: VarLenTensor, dims: Sequence[int]) -> VarLenTensor:
    view = _layout_view(inp).permute(tuple(dims))
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.select.int)
@preserve_view_inference_mode
def _select(inp: VarLenTensor, dim: int, index: int) -> VarLenTensor:
    view = _layout_view(inp).select(dim, index)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.slice.Tensor)
@preserve_view_inference_mode
def _slice(
    inp: VarLenTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> VarLenTensor:
    view = aten.slice.Tensor(_layout_view(inp), dim, start, end, step)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.narrow.default)
@preserve_view_inference_mode
def _narrow(
    inp: VarLenTensor,
    dim: int,
    start: int,
    length: int,
) -> VarLenTensor:
    view = _layout_view(inp).narrow(dim, start, length)
    return _from_layout_view(inp, view)


@VarLenTensor.implements(aten.unbind.int)
@preserve_view_inference_mode
def _unbind(inp: VarLenTensor, dim: int = 0) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(inp, view) for view in _layout_view(inp).unbind(dim)
    )


@VarLenTensor.implements(aten.split.Tensor)
@preserve_view_inference_mode
def _split(
    inp: VarLenTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(inp, view)
        for view in _layout_view(inp).split(split_size, dim)
    )


@VarLenTensor.implements(aten.split.sizes)
@VarLenTensor.implements(aten.split.default)
@VarLenTensor.implements(aten.split_with_sizes.default)
@preserve_view_inference_mode
def _split_with_sizes(
    inp: VarLenTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[VarLenTensor, ...]:
    return tuple(
        _from_layout_view(inp, view)
        for view in _layout_view(inp).split(tuple(split_sizes), dim)
    )


@VarLenTensor.implements(aten.masked_select.default)
def _masked_select(inp: VarLenTensor, mask: Tensor) -> VarLenTensor:
    return _materialize(inp, lambda x: x.masked_select(mask))


@VarLenTensor.implements(aten.index_select.default)
def _index_select(
    inp: VarLenTensor,
    dim: int,
    index: Tensor,
) -> VarLenTensor:
    return _materialize(inp, lambda x: x.index_select(dim, index))


@VarLenTensor.implements(aten.take.default)
def _take(inp: VarLenTensor, index: Tensor) -> VarLenTensor:
    return _materialize(inp, lambda x: x.take(index))


@VarLenTensor.implements(aten.index.Tensor)
def _index(
    inp: VarLenTensor,
    indices: Sequence[Tensor | None],
) -> VarLenTensor:
    return _materialize(inp, lambda x: aten.index.Tensor(x, indices))


@VarLenTensor.implements(aten.cat.default)
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


@VarLenTensor.implements(aten.stack.default)
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


def _layout_view(inp: VarLenTensor) -> Tensor:
    return torch.as_strided(
        inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )


def _from_layout_view(inp: VarLenTensor, view: Tensor) -> VarLenTensor:
    return inp.__class__(
        data=inp._data,
        offset=inp._offset,
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
    inp: VarLenTensor,
    function: Callable[[Tensor], Tensor],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> VarLenTensor:
    # Use PyTorch's own memory-format semantics to materialize data:
    start = torch.as_strided(
        inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()),
    )
    start = function(start)
    assert start.storage_offset() == 0
    assert _span_len(start.size(), start.stride()) == start.numel()

    end = torch.as_strided(
        inp._offset,
        size=inp.size(),
        stride=inp.stride(),
        storage_offset=int(inp.storage_offset()) + 1,
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

    return inp.__class__(
        data=inp._data[index].to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        ),
        offset=offset.to(device, non_blocking=non_blocking),
        size=size,
        stride=stride,
        storage_offset=0,
    )
