from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex, cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from typing_extensions import Self, override

from sdm.tensor import StringTensor
from sdm.tensor.io import arrow_as_tensor, to_arrow, to_cudf
from sdm.tensor.io.arrow import _combine_arrow_chunks
from sdm.tensor.mixin import _resolve_device

if TYPE_CHECKING:
    import cudf

aten = torch.ops.aten


def preserve_view_inference_mode(fn: Callable) -> Callable:
    r"""Preserve input inference state for tensor view operations."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(args[0].is_inference()):
            return fn(*args, **kwargs)

    return wrapper


class ColumnarTensor(Tensor):
    r"""A :class:`torch.Tensor` for column-wise heterogeneous data.

    A :class:`ColumnarTensor` exposes a tensor-centric interface for columnar
    data with shape ``[..., C]``.
    Each of the ``C`` columns is stored independently and may use a different
    tensor subclass or dtype.

    Args:
        columns: Per-column values.
        validity: Per-column validity bitmap.
        size: The shape of the tensor ``[...]``.
        device: The device.
            value. ``None`` denotes an entirely valid column.
    """

    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _columns: tuple[Tensor, ...]
    _validity: tuple[Tensor | None, ...]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        self,
        columns: Sequence[Tensor],
        validity: Sequence[Tensor | None] | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        pass

    def __new__(
        cls,
        columns: Sequence[Tensor],
        validity: Sequence[Tensor | None] | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor wrapper."""
        # Avoid a circular import through `sdm.tensor`.
        from sdm.tensor import (  # noqa: PLC0415
            CategoricalTensor,
            TableTensor,
        )

        if size is not None and len(size) == 0:
            raise ValueError("Expected 'size' to be non-empty")

        for i, column in enumerate(columns):
            if isinstance(
                column, CategoricalTensor | ColumnarTensor | TableTensor
            ):
                raise TypeError(
                    f"Expected value {i} in {cls.__name__!r} to be a single "
                    f"column tensor (got {column.__class__.__name__!r})"
                )

        if validity is None:
            validity = (None,) * len(columns)
        if len(validity) != len(columns):
            raise ValueError(
                f"Expected 'validity' to contain {len(columns)} entries "
                f"(got {len(validity)})"
            )

        columns = tuple(columns)
        validity = tuple(validity)
        size = tuple(size) if size is not None else size
        device = _resolve_device(device)

        for i, (column, valid) in enumerate(zip(columns, validity)):
            size = tuple(column.size()) if size is None else size
            device = column.device if device is None else device

            if column.size() != size:
                raise ValueError(
                    f"Expected value {i} in {cls.__name__!r} to have size "
                    f"{size} (got {tuple(column.size())})"
                )
            if column.device != device:
                raise ValueError(
                    f"Expected value {i} in {cls.__name__!r} to be on "
                    f"device '{device}' (got '{column.device}')"
                )

            if valid is not None and valid.dtype != torch.bool:
                raise TypeError(
                    f"Expected validity mask {i} in {cls.__name__!r} to have "
                    f"dtype '{torch.bool}' (got '{valid.dtype}')"
                )
            if valid is not None and valid.size() != size:
                raise ValueError(
                    f"Expected validity mask {i} in {cls.__name__!r} to have "
                    f"size {size} (got {tuple(valid.size())})"
                )
            if valid is not None and valid.device != device:
                raise ValueError(
                    f"Expected validity mask {i} in {cls.__name__!r} to be on "
                    f"device '{device}' (got '{valid.device}')"
                )

        if size is None:
            raise ValueError(
                "Expected 'size' to be given for zero columnar data"
            )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=(*size, len(columns)),
            dtype=torch.uint8,  # NOTE Do not use.
            device=device,
            requires_grad=False,
        )

        out._columns = columns
        out._validity = validity

        return out

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a :class:`pyarrow.Array`.

        Args:
            array: The :class:`pyarrow.Array` or
                :class:`pyarrow.ChunkedArray`.
            device: The device.
        """
        device = torch.device("cpu" if device is None else device)

        if isinstance(array, pa.ChunkedArray):
            array = _combine_arrow_chunks(array)

        valid = None
        if array.null_count > 0:
            valid = arrow_as_tensor(
                array.is_valid(),
                dtype=torch.bool,
                device=device,
            )

        is_string = pa.types.is_string(array.type)
        is_large_string = pa.types.is_large_string(array.type)
        if is_string or is_large_string:
            column = StringTensor.from_arrow(array, device=device)
        else:
            if valid is not None and pa.types.is_integer(array.type):
                array = pc.fill_null(array, pa.scalar(0, type=array.type))
            column = arrow_as_tensor(array, device=device)

        return cls(
            columns=(column,),
            validity=(valid,),
            device=device,
        )

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series | cudf.Index,
        *,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create tensor from a :class:`cudf.Series`.

        Args:
            ser: The :class:`cudf.Series` or :class:`cudf.Index`.
            device: The device.
        """
        from cudf.api.types import is_integer_dtype, is_string_dtype

        valid = None
        if ser._column.null_count > 0:
            valid = torch.from_dlpack(ser.notna().to_dlpack())
            valid = valid.to(device=device, dtype=torch.bool)

        if is_string_dtype(ser.dtype):
            column = StringTensor.from_cudf(ser, device=device)
        else:
            if valid is not None and is_integer_dtype(ser.dtype):
                ser = ser.fillna(0)
            column = torch.from_dlpack(ser.to_dlpack()).to(device)

        return cls(
            columns=(column,),
            validity=(valid,),
            device=device,
        )

    def to_arrow(self, names: Sequence[str] | None = None) -> pa.Table:
        r"""Convert this tensor to a two-dimensional :class:`pyarrow.Table`.

        Args:
            names: The column names.
        """
        if names is None:
            names = tuple(str(i) for i in range(self.size(-1)))
        elif len(names) != self.size(-1):
            raise ValueError(
                f"Expected 'names' to contain {self.size(-1)} entries "
                f"(got {len(names)})"
            )

        return pa.Table.from_arrays(
            arrays=[
                to_arrow(column, valid)
                for column, valid in zip(self._columns, self._validity)
            ],
            names=names,
        )

    def to_cudf(self, names: Sequence[str] | None = None) -> cudf.DataFrame:
        r"""Convert this tensor to a two-dimensional :class:`cudf.DataFrame`.

        Args:
            names: The column names.
        """
        import cudf

        if names is None:
            names = tuple(str(i) for i in range(self.size(-1)))
        elif len(names) != self.size(-1):
            raise ValueError(
                f"Expected 'names' to contain {self.size(-1)} entries "
                f"(got {len(names)})"
            )

        return cudf.DataFrame(
            {
                name: to_cudf(column, valid)
                for name, column, valid in zip(
                    names,
                    self._columns,
                    self._validity,
                )
            },
        )

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

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (
            self._columns,
            self._validity,
            tuple(self.size())[:-1],
            self.device,
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
            return handler(*args, **(kwargs or {}))

        raise NotImplementedError(
            f"'{func}' is not supported for {cls.__name__!r}"
        )

    @override
    def is_shared(self) -> bool:
        return all(column.is_shared() for column in self._columns) and all(
            valid is None or valid.is_shared() for valid in self._validity
        )

    @override
    def share_memory_(self) -> Self:
        for column in self._columns:
            column.share_memory_()
        for valid in self._validity:
            if valid is not None:
                valid.share_memory_()
        return self

    @override
    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            column.is_contiguous(memory_format=memory_format)
            for column in self._columns
        ) and all(
            valid is None or valid.is_contiguous(memory_format=memory_format)
            for valid in self._validity
        )

    @override
    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Self:
        if self.is_contiguous(memory_format=memory_format):
            return self
        return _contiguous(self, memory_format=memory_format)

    @override
    def tolist(self) -> Any:
        def reshape(values: list[Any], size: tuple[int, ...]) -> Any:
            if len(size) == 0:
                return values[0]

            step = math.prod(size[1:])
            return [
                reshape(values[i * step : (i + 1) * step], size[1:])
                for i in range(size[0])
            ]

        def columns_to_rows(
            columns: Sequence[Any],
            size: tuple[int, ...],
        ) -> Any:
            if len(size) == 0:
                return list(columns)

            return [
                columns_to_rows(
                    columns=[column[i] for column in columns],
                    size=size[1:],
                )
                for i in range(size[0])
            ]

        size = tuple(self.size()[:-1])
        columns = [
            reshape(to_arrow(column, valid).to_pylist(), size)
            for column, valid in zip(self._columns, self._validity)
        ]
        return columns_to_rows(columns, size=tuple(self.size()[:-1]))

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out


@ColumnarTensor.implements(aten.isnan.default)
def _isnan(inp: ColumnarTensor) -> Tensor:
    if inp.numel() == 0:
        return torch.empty(inp.size(), dtype=torch.bool, device=inp.device)

    masks = []
    for column, valid in zip(inp._columns, inp._validity):
        mask: Tensor | None = None
        if column.is_floating_point():
            mask = column.isnan()
        if valid is not None:
            mask = ~valid if mask is None else mask | ~valid
        if mask is None:
            mask = torch.zeros(
                column.size(),
                dtype=torch.bool,
                device=column.device,
            )
        masks.append(mask)
    return torch.stack(masks, dim=-1)


@ColumnarTensor.implements(aten.alias.default)
@preserve_view_inference_mode
def _alias(inp: ColumnarTensor) -> ColumnarTensor:
    return inp.__class__(
        columns=inp._columns,
        validity=inp._validity,
        size=inp.size()[:-1],
        device=inp.device,
    )


@ColumnarTensor.implements(aten._to_copy.default)
def _to_copy(
    inp: ColumnarTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:

    # Wrapper dtype is a placeholder, so same dtype means no conversion:
    if dtype == inp.dtype:
        dtype = None

    if dtype is not None:
        raise TypeError(
            f"Can't convert {inp.__class__.__name__!r} to dtype '{dtype}'"
        )

    return inp.__class__(
        columns=[
            aten._to_copy.default(
                column,
                device=device,
                dtype=None,
                layout=layout,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )
            for column in inp._columns
        ],
        validity=[
            None
            if valid is None
            else aten._to_copy.default(
                valid,
                device=device,
                dtype=None,
                layout=layout,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )
            for valid in inp._validity
        ],
        size=inp.size()[:-1],
        device=device,
    )


@ColumnarTensor.implements(aten.clone.default)
def _clone(
    inp: ColumnarTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> ColumnarTensor:
    out = _to_copy(inp, memory_format=memory_format)
    assert isinstance(out, ColumnarTensor)
    return out


@ColumnarTensor.implements(aten.contiguous.default)
def _contiguous(
    inp: ColumnarTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> ColumnarTensor:
    return inp.__class__(
        columns=[
            column.contiguous(memory_format=memory_format)
            for column in inp._columns
        ],
        validity=[
            None
            if valid is None
            else valid.contiguous(memory_format=memory_format)
            for valid in inp._validity
        ],
        size=inp.size()[:-1],
        device=inp.device,
    )


@ColumnarTensor.implements(aten.is_pinned.default)
def _is_pinned(inp: ColumnarTensor) -> bool:
    return all(column.is_pinned() for column in inp._columns) and all(
        valid is None or valid.is_pinned() for valid in inp._validity
    )


@ColumnarTensor.implements(aten._pin_memory.default)
def _pin_memory(inp: ColumnarTensor) -> ColumnarTensor:
    return inp.__class__(
        columns=[column.pin_memory() for column in inp._columns],
        validity=[
            None if valid is None else valid.pin_memory()
            for valid in inp._validity
        ],
        size=inp.size()[:-1],
        device=inp.device,
    )


@ColumnarTensor.implements(aten.equal.default)
def _equal(inp: ColumnarTensor, other: Tensor) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    for column1, column2 in zip(inp._columns, other._columns):
        if not column1.equal(column2):
            return False

    for valid1, valid2 in zip(inp._validity, other._validity):
        if valid1 is None and valid2 is not None and not valid2.all():
            return False
        if valid2 is None and valid1 is not None and not valid1.all():
            return False
        if (
            valid1 is not None
            and valid2 is not None
            and not valid1.equal(valid2)
        ):
            return False

    return True


@ColumnarTensor.implements(aten.allclose.default)
def _allclose(
    inp: ColumnarTensor,
    other: Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if inp.__class__ is not other.__class__:
        return False
    if inp.size() != other.size():
        return False

    for column1, column2 in zip(inp._columns, other._columns):
        if column1.is_floating_point() and column2.is_floating_point():
            if not column1.allclose(column2, rtol, atol, equal_nan):
                return False
        elif not column1.equal(column2):
            return False

    for valid1, valid2 in zip(inp._validity, other._validity):
        if valid1 is None and valid2 is not None and not valid2.all():
            return False
        if valid2 is None and valid1 is not None and not valid1.all():
            return False
        if (
            valid1 is not None
            and valid2 is not None
            and not valid1.equal(valid2)
        ):
            return False

    return True


@ColumnarTensor.implements(aten.view.default)
@preserve_view_inference_mode
def _view(inp: ColumnarTensor, size: Sequence[int]) -> ColumnarTensor:
    size = tuple(size)
    for i, dim_size in enumerate(size):
        if dim_size < -1:
            raise RuntimeError(
                f"Invalid shape dimension {dim_size} at index {i} of shape "
                f"{size}"
            )

    if size.count(-1) > 1:
        raise RuntimeError("Only one dimension can be inferred")

    if -1 in size:
        known = math.prod(dim_size for dim_size in size if dim_size != -1)
        if known == 0:
            raise RuntimeError(
                f"Cannot reshape tensor of {inp.numel()} elements into "
                f"shape {size} because the unspecified dimension size -1 can "
                f"be any value and is ambiguous"
            )
        if inp.numel() % known != 0:
            raise RuntimeError(
                f"Shape {size} is invalid for input of size {inp.numel()}"
            )
        dim = size.index(-1)
        size = (*size[:dim], inp.numel() // known, *size[dim + 1 :])

    if len(size) == 0 or size[-1] != inp.size(-1):
        _columns = "column" if inp.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't reshape {inp.__class__.__name__!r} with "
            f"{inp.size(-1)} {_columns} into shape {size}"
        )

    return inp.__class__(
        columns=[column.view(size[:-1]) for column in inp._columns],
        validity=[
            None if valid is None else valid.view(size[:-1])
            for valid in inp._validity
        ],
        size=size[:-1],
        device=inp.device,
    )


@ColumnarTensor.implements(aten._unsafe_view.default)
@preserve_view_inference_mode
def _unsafe_view(
    inp: ColumnarTensor,
    size: Sequence[int],
) -> ColumnarTensor:
    return _view(inp, size)


@ColumnarTensor.implements(aten.squeeze.default)
@preserve_view_inference_mode
def _squeeze(inp: ColumnarTensor) -> ColumnarTensor:
    return _squeeze_dims(inp, range(inp.dim() - 1))


@ColumnarTensor.implements(aten.squeeze.dim)
@preserve_view_inference_mode
def _squeeze_dim(inp: ColumnarTensor, dim: int) -> ColumnarTensor:
    return _squeeze_dims(inp, (dim,))


@ColumnarTensor.implements(aten.squeeze.dims)
@preserve_view_inference_mode
def _squeeze_dims(inp: ColumnarTensor, dim: Sequence[int]) -> ColumnarTensor:
    dims = tuple(_normalize_dim(inp, d) for d in dim)

    if any(dim == inp.dim() - 1 for dim in dims):
        raise RuntimeError(
            f"Can't squeeze the column dimension of {inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=[column.squeeze(dims) for column in inp._columns],
        validity=[
            None if valid is None else valid.squeeze(dims)
            for valid in inp._validity
        ],
        size=tuple(
            dim_size
            for i, dim_size in enumerate(inp.size()[:-1])
            if i not in dims or dim_size != 1
        ),
        device=inp.device,
    )


@ColumnarTensor.implements(aten.unsqueeze.default)
@preserve_view_inference_mode
def _unsqueeze(inp: ColumnarTensor, dim: int) -> ColumnarTensor:
    if dim < -inp.dim() - 1 or dim > inp.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.dim() - 1}, {inp.dim()}], but got {dim})"
        )
    dim = dim % (inp.dim() + 1)

    if dim == inp.dim():
        raise RuntimeError(
            f"Can't unsqueeze after the column dimension of "
            f"{inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=[column.unsqueeze(dim) for column in inp._columns],
        validity=[
            None if valid is None else valid.unsqueeze(dim)
            for valid in inp._validity
        ],
        size=(*inp.size()[:dim], 1, *inp.size()[dim:-1]),
        device=inp.device,
    )


@ColumnarTensor.implements(aten.expand.default)
@preserve_view_inference_mode
def _expand(
    inp: ColumnarTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> ColumnarTensor:
    size = tuple(size)

    if len(size) < inp.dim():
        raise RuntimeError(
            f"expand: the number of sizes provided ({len(size)}) must be "
            f"greater or equal to the number of dimensions in the tensor "
            f"({inp.dim()})"
        )

    if size[-1] not in (-1, inp.size(-1)):
        _columns = "column" if inp.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't expand {inp.__class__.__name__!r} with "
            f"{inp.size(-1)} {_columns} to shape {size}"
        )

    old_size = (*(1,) * (len(size) - inp.dim()), *inp.size())

    return inp.__class__(
        columns=[
            aten.expand.default(
                column,
                size[:-1],
                implicit=implicit,
            )
            for column in inp._columns
        ],
        validity=[
            None
            if valid is None
            else aten.expand.default(
                valid,
                size[:-1],
                implicit=implicit,
            )
            for valid in inp._validity
        ],
        size=tuple(
            old if new == -1 else new for old, new in zip(old_size, size)
        )[:-1],
        device=inp.device,
    )


@ColumnarTensor.implements(aten.transpose.int)
@preserve_view_inference_mode
def _transpose(
    inp: ColumnarTensor,
    dim0: int,
    dim1: int,
) -> ColumnarTensor:

    dim0 = _normalize_dim(inp, dim0)
    dim1 = _normalize_dim(inp, dim1)

    if dim0 == dim1 == inp.dim() - 1:
        return _alias(inp)

    if inp.dim() - 1 in (dim0, dim1):
        raise RuntimeError(
            f"Can't transpose the column dimension of "
            f"{inp.__class__.__name__!r}"
        )

    size = list(inp.size())
    size[dim0], size[dim1] = size[dim1], size[dim0]

    return inp.__class__(
        columns=[column.transpose(dim0, dim1) for column in inp._columns],
        validity=[
            None if valid is None else valid.transpose(dim0, dim1)
            for valid in inp._validity
        ],
        size=size[:-1],
        device=inp.device,
    )


@ColumnarTensor.implements(aten.permute.default)
@preserve_view_inference_mode
def _permute(inp: ColumnarTensor, dims: Sequence[int]) -> ColumnarTensor:
    dims = tuple(_normalize_dim(inp, dim) for dim in dims)
    if dims[-1] != inp.dim() - 1:
        raise RuntimeError(
            f"Can't permute the column dimension of {inp.__class__.__name__!r}"
        )

    return inp.__class__(
        columns=[column.permute(dims[:-1]) for column in inp._columns],
        validity=[
            None if valid is None else valid.permute(dims[:-1])
            for valid in inp._validity
        ],
        size=tuple(inp.size(dim) for dim in dims[:-1]),
        device=inp.device,
    )


@ColumnarTensor.implements(aten.select.int)
@preserve_view_inference_mode
def _select(inp: ColumnarTensor, dim: int, index: int) -> Tensor:
    dim = _normalize_dim(inp, dim)

    if dim == inp.dim() - 1:
        if inp._validity[index] is not None:
            raise RuntimeError(
                f"Can't select a nullable column from "
                f"{inp.__class__.__name__!r} without losing its validity mask"
            )
        return aten.alias.default(inp._columns[index])

    return inp.__class__(
        columns=[column.select(dim, index) for column in inp._columns],
        validity=[
            None if valid is None else valid.select(dim, index)
            for valid in inp._validity
        ],
        size=(*inp.size()[:dim], *inp.size()[dim + 1 : -1]),
        device=inp.device,
    )


@ColumnarTensor.implements(aten.slice.Tensor)
@preserve_view_inference_mode
def _slice(
    inp: ColumnarTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> ColumnarTensor:
    dim = _normalize_dim(inp, dim)

    if dim == inp.dim() - 1:
        return inp.__class__(
            columns=inp._columns[slice(start, end, step)],
            size=inp.size()[:-1],
            device=inp.device,
            validity=inp._validity[slice(start, end, step)],
        )

    return inp.__class__(
        columns=[
            aten.slice.Tensor(column, dim, start, end, step)
            for column in inp._columns
        ],
        validity=[
            None
            if valid is None
            else aten.slice.Tensor(valid, dim, start, end, step)
            for valid in inp._validity
        ],
        size=(
            *inp.size()[:dim],
            len(range(inp.size(dim))[slice(start, end, step)]),
            *inp.size()[dim + 1 : -1],
        ),
        device=inp.device,
    )


@ColumnarTensor.implements(aten.narrow.default)
@preserve_view_inference_mode
def _narrow(
    inp: ColumnarTensor,
    dim: int,
    start: int,
    length: int,
) -> ColumnarTensor:
    return _slice(inp, dim=dim, start=start, end=start + length)


@ColumnarTensor.implements(aten.unbind.int)
@preserve_view_inference_mode
def _unbind(inp: ColumnarTensor, dim: int = 0) -> tuple[Tensor, ...]:
    dim = _normalize_dim(inp, dim)

    if dim == inp.dim() - 1:
        if any(valid is not None for valid in inp._validity):
            raise RuntimeError(
                f"Can't unbind nullable columns from "
                f"{inp.__class__.__name__!r} without losing validity masks"
            )
        return tuple(aten.alias.default(column) for column in inp._columns)

    columns_list = [column.unbind(dim) for column in inp._columns]
    if len(columns_list) == 0:
        size = (*inp.size()[:dim], *inp.size()[dim + 1 : -1])
        return tuple(
            inp.__class__(
                columns=(),
                size=size,
                device=inp.device,
            )
            for _ in range(inp.size(dim))
        )

    validity_list = [
        None if valid is None else valid.unbind(dim) for valid in inp._validity
    ]
    return tuple(
        inp.__class__(
            columns=columns,
            device=inp.device,
            validity=[
                None if validity is None else validity[i]
                for validity in validity_list
            ],
        )
        for i, columns in enumerate(zip(*columns_list))
    )


@ColumnarTensor.implements(aten.split.Tensor)
@preserve_view_inference_mode
def _split(
    inp: ColumnarTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[ColumnarTensor, ...]:
    split_sizes = tuple(
        min(split_size, inp.size(dim) - start)
        for start in range(0, inp.size(dim), split_size)
    )
    return _split_with_sizes(inp, split_sizes, dim=dim)


@ColumnarTensor.implements(aten.split.sizes)
@ColumnarTensor.implements(aten.split.default)
@ColumnarTensor.implements(aten.split_with_sizes.default)
@preserve_view_inference_mode
def _split_with_sizes(
    inp: ColumnarTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[ColumnarTensor, ...]:
    dim = _normalize_dim(inp, dim)
    split_sizes = tuple(split_sizes)

    if dim == inp.dim() - 1:
        if sum(split_sizes) != inp.size(dim):
            raise RuntimeError(
                f"split_with_sizes expects split_sizes to sum exactly to "
                f"{inp.size(dim)} (input tensor's size at dimension {dim}), "
                f"but got split_sizes={split_sizes}"
            )

        start = 0
        outs = []
        for split_size in split_sizes:
            out = inp.__class__(
                columns=inp._columns[start : start + split_size],
                validity=inp._validity[start : start + split_size],
                size=inp.size()[:-1],
                device=inp.device,
            )
            start += split_size
            outs.append(out)
        return tuple(outs)

    columns_list = [col.split(split_sizes, dim=dim) for col in inp._columns]
    if len(columns_list) == 0:
        return tuple(
            inp.__class__(
                columns=(),
                size=(
                    *inp.size()[:dim],
                    split_size,
                    *inp.size()[dim + 1 : -1],
                ),
                device=inp.device,
            )
            for split_size in split_sizes
        )

    validity_list = [
        None if valid is None else valid.split(split_sizes, dim=dim)
        for valid in inp._validity
    ]
    return tuple(
        inp.__class__(
            columns=columns,
            validity=[
                None if validity is None else validity[i]
                for validity in validity_list
            ],
            device=inp.device,
        )
        for i, columns in enumerate(zip(*columns_list))
    )


@ColumnarTensor.implements(aten.index_select.default)
def _index_select(
    inp: ColumnarTensor,
    dim: int,
    index: Tensor,
) -> ColumnarTensor:
    dim = _normalize_dim(inp, dim)
    if dim == inp.dim() - 1:
        column_indices = index.tolist()
        return inp.__class__(
            columns=[inp._columns[i] for i in column_indices],
            validity=[inp._validity[i] for i in column_indices],
            size=inp.size()[:-1],
            device=inp.device,
        )

    return inp.__class__(
        columns=[column.index_select(dim, index) for column in inp._columns],
        validity=[
            None if valid is None else valid.index_select(dim, index)
            for valid in inp._validity
        ],
        size=(
            *inp.size()[:dim],
            index.numel(),
            *inp.size()[dim + 1 : -1],
        ),
        device=inp.device,
    )


@ColumnarTensor.implements(aten.index.Tensor)
def _index(
    inp: ColumnarTensor,
    indices: Sequence[Tensor | None],
) -> ColumnarTensor:
    current_dim = 0
    has_other_index = False
    column_index: Tensor | None = None
    for index in indices:
        if index is None:
            current_dim += 1
            continue

        # Check whether we index the column dimension:
        num_indexed_dims = index.dim() if index.dtype == torch.bool else 1
        if current_dim <= inp.dim() - 1 < current_dim + num_indexed_dims:
            if num_indexed_dims != 1:
                raise RuntimeError(
                    f"Can't index the column dimension of "
                    f"{inp.__class__.__name__!r} with a multi-dimensional "
                    f"index"
                )
            column_index = index
        else:
            has_other_index = True
        current_dim += num_indexed_dims

    if column_index is not None:
        if has_other_index or column_index.dim() != 1:
            raise RuntimeError(
                f"Can't index the column dimension of "
                f"{inp.__class__.__name__!r} together with other dimensions"
            )
        if column_index.dtype == torch.bool:
            column_index = column_index.nonzero().view(-1)
        column_indices = column_index.tolist()
        return inp.__class__(
            columns=[inp._columns[i] for i in column_indices],
            size=inp.size()[:-1],
            device=inp.device,
            validity=[inp._validity[i] for i in column_indices],
        )

    if len(inp._columns) == 0:
        dummy = torch.empty(inp.size(), device=inp.device)
        return inp.__class__(
            columns=(),
            size=aten.index.Tensor(dummy, indices).size()[:-1],
            device=inp.device,
        )

    return inp.__class__(
        columns=[
            aten.index.Tensor(column, indices) for column in inp._columns
        ],
        validity=[
            None if valid is None else aten.index.Tensor(valid, indices)
            for valid in inp._validity
        ],
        device=inp.device,
    )


@ColumnarTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> ColumnarTensor:
    if not all(isinstance(tensor, ColumnarTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be {ColumnarTensor.__name__!r} instances"
        )

    tensors = cast(Sequence[ColumnarTensor], tensors)
    dim = _normalize_dim(tensors[0], dim)

    if dim == tensors[0].dim() - 1:
        return tensors[0].__class__(
            columns=tuple(chain.from_iterable(t._columns for t in tensors)),
            validity=tuple(chain.from_iterable(t._validity for t in tensors)),
            size=tensors[0].size()[:-1],
            device=tensors[0].device,
        )

    return tensors[0].__class__(
        columns=[
            torch.cat([tensor._columns[i] for tensor in tensors], dim=dim)
            for i in range(tensors[0].size(-1))
        ],
        validity=[
            _combine_validity(tensors, column=i, dim=dim, stack=False)
            for i in range(tensors[0].size(-1))
        ],
        size=(
            *tensors[0].size()[:dim],
            sum(tensor.size(dim) for tensor in tensors),
            *tensors[0].size()[dim + 1 : -1],
        ),
        device=tensors[0].device,
    )


@ColumnarTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> ColumnarTensor:
    if not all(isinstance(tensor, ColumnarTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be {ColumnarTensor.__name__!r} instances"
        )

    tensors = cast(Sequence[ColumnarTensor], tensors)

    if dim < -tensors[0].dim() - 1 or dim > tensors[0].dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-tensors[0].dim() - 1}, {tensors[0].dim()}], but got {dim})"
        )
    dim = dim % (tensors[0].dim() + 1)

    if dim >= tensors[0].dim():
        raise RuntimeError(
            f"Can't stack after the column dimension of "
            f"{tensors[0].__class__.__name__!r}"
        )

    return tensors[0].__class__(
        columns=[
            torch.stack([tensor._columns[i] for tensor in tensors], dim=dim)
            for i in range(tensors[0].size(-1))
        ],
        validity=[
            _combine_validity(tensors, column=i, dim=dim, stack=True)
            for i in range(tensors[0].size(-1))
        ],
        size=(
            *tensors[0].size()[:dim],
            len(tensors),
            *tensors[0].size()[dim:-1],
        ),
        device=tensors[0].device,
    )


# Helpers #####################################################################


def _combine_validity(
    tensors: Sequence[ColumnarTensor],
    column: int,
    dim: int,
    *,
    stack: bool,
) -> Tensor | None:
    if all(tensor._validity[column] is None for tensor in tensors):
        return None

    masks: list[Tensor] = []
    for tensor in tensors:
        valid = tensor._validity[column]
        if valid is None:
            valid = torch.ones(
                tensor.size()[:-1],
                dtype=torch.bool,
                device=tensor.device,
            )
        masks.append(valid)

    if stack:
        return torch.stack(masks, dim=dim)
    return torch.cat(masks, dim=dim)


def _normalize_dim(inp: Tensor, dim: int) -> int:
    if dim < -inp.dim() or dim >= inp.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.dim()}, {inp.dim() - 1}], but got {dim})"
        )
    return dim % inp.dim()
