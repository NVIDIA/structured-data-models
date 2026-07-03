import math
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, ClassVar, SupportsIndex, TypeVar, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import override

from sdm.tensor import StringTensor, VarLenTensor
from sdm.tensor.io import to_arrow

aten = torch.ops.aten

SelfColumnarTensor = TypeVar(
    "SelfColumnarTensor",
    bound="ColumnarTensor",
)


class ColumnarTensor(Tensor):
    r"""A :class:`torch.Tensor` for column-wise heterogeneous data.

    A :class:`ColumnarTensor` exposes a tensor-centric interface for columnar
    data with shape ``[..., C]``.
    Each of the ``C`` columns is stored independently and may use a different
    tensor subclass or dtype.

    Args:
        columns: Per-column values.
        size: The shape of the tensor ``[...]``.
        device: The device.
    """

    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _columns: tuple[Tensor, ...]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        self,
        columns: Sequence[Tensor],
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        pass

    def __new__(
        cls: type[SelfColumnarTensor],
        columns: Sequence[Tensor],
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> SelfColumnarTensor:
        r"""Create a tensor wrapper."""
        from sdm.tensor import CategoricalTensor, TableTensor

        if size is not None and len(size) == 0:
            raise ValueError("Expected 'size' to be non-empty")

        for i, column in enumerate(columns):
            if isinstance(
                column, CategoricalTensor | ColumnarTensor | TableTensor
            ):
                raise TypeError(
                    f"Expected value {i} in '{cls.__name__}' to be a single "
                    f"column tensor (got '{column.__class__.__name__}')"
                )
        columns = tuple(columns)
        size = tuple(size) if size is not None else size
        device = torch.device(device) if device is not None else None

        for i, column in enumerate(columns):
            size = tuple(column.size()) if size is None else size
            device = column.device if device is None else device

            if column.size() != size:
                raise ValueError(
                    f"Expected value {i} in '{cls.__name__}' to have size "
                    f"{size} (got {tuple(column.size())})"
                )

            if column.device != device:
                raise ValueError(
                    f"Expected value {i} in '{cls.__name__}' to be on "
                    f"device '{device}' (got '{column.device}')"
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

        return out

    @classmethod
    def from_arrow(
        cls: type[SelfColumnarTensor],
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | str | None = None,
    ) -> SelfColumnarTensor:
        r"""Create tensor from a ``pyarrow`` array.

        Args:
            array: The ``pyarrow`` array.
            device: The device.
        """
        device = torch.device("cpu" if device is None else device)

        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

        is_string = pa.types.is_string(array.type)
        is_large_string = pa.types.is_large_string(array.type)
        if is_string or is_large_string:
            column = StringTensor.from_arrow(array, device=device)
        else:
            if array.null_count > 0 and pa.types.is_integer(array.type):
                raise ValueError(
                    f"'{cls.__name__}' cannot represent null integer values"
                )
            values = array.to_numpy(
                zero_copy_only=False,
                writable=device.type == "cpu",
            )
            column = torch.as_tensor(values, device=device)

        return cls(columns=(column,), device=device)

    def to_arrow(self, columns: Sequence[str] | None = None) -> pa.Table:
        r"""Convert this tensor to a flat ``pyarrow`` table.

        Args:
            columns: The column names.
        """
        if columns is None:
            columns = tuple(str(i) for i in range(self.size(-1)))
        elif len(columns) != self.size(-1):
            raise ValueError(
                f"Expected 'columns' to contain {self.size(-1)} entries "
                f"(got {len(columns)})"
            )

        arrays = [
            column.to_arrow()
            if isinstance(column, VarLenTensor)
            else to_arrow(column)
            for column in self.unbind(-1)
        ]
        return pa.Table.from_arrays(arrays, names=columns)

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        r"""Register a ``__torch_dispatch__`` implementation."""
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (self._columns, tuple(self.size())[:-1], self.device)
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
            f"'{func}' is not supported for '{cls.__name__}'"
        )

    @override
    def is_shared(self) -> bool:
        return all(column.is_shared() for column in self._columns)

    @override
    def share_memory_(self) -> "ColumnarTensor":
        for column in self._columns:
            column.share_memory_()
        return self

    @override
    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            column.is_contiguous(memory_format=memory_format)
            for column in self._columns
        )

    @override
    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> "ColumnarTensor":
        if self.is_contiguous(memory_format=memory_format):
            return self
        return _contiguous(self, memory_format=memory_format)

    @override
    def tolist(self) -> Any:
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

        columns = [column.tolist() for column in self._columns]
        return columns_to_rows(columns, size=tuple(self.size()[:-1]))

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        out = f"{self.__class__.__name__}("
        out += f"size={tuple(self.size())}"
        if not self.is_cpu:
            out += f", device={self.device}"
        out += ")"
        return out


@ColumnarTensor.implements(aten.alias.default)
def _alias(input: ColumnarTensor) -> ColumnarTensor:
    return input.__class__(
        columns=input._columns,
        size=input.size()[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten._to_copy.default)
def _to_copy(
    input: ColumnarTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> Tensor:

    if dtype is not None:
        raise TypeError(
            f"Can't convert '{input.__class__.__name__}' to dtype '{dtype}'"
        )

    return input.__class__(
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
            for column in input._columns
        ],
        size=input.size()[:-1],
        device=device,
    )


@ColumnarTensor.implements(aten.clone.default)
def _clone(
    input: ColumnarTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> ColumnarTensor:
    out = _to_copy(input, memory_format=memory_format)
    assert isinstance(out, ColumnarTensor)
    return out


@ColumnarTensor.implements(aten.contiguous.default)
def _contiguous(
    input: ColumnarTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> ColumnarTensor:
    return input.__class__(
        columns=[
            column.contiguous(memory_format=memory_format)
            for column in input._columns
        ],
        size=input.size()[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.is_pinned.default)
def _is_pinned(input: ColumnarTensor) -> bool:
    return all(column.is_pinned() for column in input._columns)


@ColumnarTensor.implements(aten._pin_memory.default)
def _pin_memory(input: ColumnarTensor) -> ColumnarTensor:
    return input.__class__(
        columns=[column.pin_memory() for column in input._columns],
        size=input.size()[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.view.default)
def _view(input: ColumnarTensor, size: Sequence[int]) -> ColumnarTensor:
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
                f"Cannot reshape tensor of {input.numel()} elements into "
                f"shape {size} because the unspecified dimension size -1 can "
                f"be any value and is ambiguous"
            )
        if input.numel() % known != 0:
            raise RuntimeError(
                f"Shape {size} is invalid for input of size {input.numel()}"
            )
        dim = size.index(-1)
        size = (*size[:dim], input.numel() // known, *size[dim + 1 :])

    if len(size) == 0 or size[-1] != input.size(-1):
        _columns = "column" if input.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't reshape '{input.__class__.__name__}' with "
            f"{input.size(-1)} {_columns} into shape {size}"
        )

    return input.__class__(
        columns=[column.view(size[:-1]) for column in input._columns],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten._unsafe_view.default)
def _unsafe_view(
    input: ColumnarTensor,
    size: Sequence[int],
) -> ColumnarTensor:
    return _view(input, size)


@ColumnarTensor.implements(aten.squeeze.default)
def _squeeze(input: ColumnarTensor) -> ColumnarTensor:
    return _squeeze_dims(input, range(input.dim() - 1))


@ColumnarTensor.implements(aten.squeeze.dim)
def _squeeze_dim(input: ColumnarTensor, dim: int) -> ColumnarTensor:
    return _squeeze_dims(input, (dim,))


@ColumnarTensor.implements(aten.squeeze.dims)
def _squeeze_dims(input: ColumnarTensor, dim: Sequence[int]) -> ColumnarTensor:
    dims = tuple(_normalize_dim(input, d) for d in dim)

    if any(dim == input.dim() - 1 for dim in dims):
        raise RuntimeError(
            f"Can't squeeze the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=[column.squeeze(dims) for column in input._columns],
        size=tuple(
            dim_size
            for i, dim_size in enumerate(input.size()[:-1])
            if i not in dims or dim_size != 1
        ),
        device=input.device,
    )


@ColumnarTensor.implements(aten.unsqueeze.default)
def _unsqueeze(input: ColumnarTensor, dim: int) -> ColumnarTensor:
    if dim < -input.dim() - 1 or dim > input.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.dim() - 1}, {input.dim()}], but got {dim})"
        )
    dim = dim % (input.dim() + 1)

    if dim == input.dim():
        raise RuntimeError(
            f"Can't unsqueeze after the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=[column.unsqueeze(dim) for column in input._columns],
        size=(*input.size()[:dim], 1, *input.size()[dim:-1]),
        device=input.device,
    )


@ColumnarTensor.implements(aten.expand.default)
def _expand(
    input: ColumnarTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> ColumnarTensor:
    size = tuple(size)

    if len(size) < input.dim():
        raise RuntimeError(
            f"expand: the number of sizes provided ({len(size)}) must be "
            f"greater or equal to the number of dimensions in the tensor "
            f"({input.dim()})"
        )

    if size[-1] not in (-1, input.size(-1)):
        _columns = "column" if input.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't expand '{input.__class__.__name__}' with "
            f"{input.size(-1)} {_columns} to shape {size}"
        )

    old_size = (*(1,) * (len(size) - input.dim()), *input.size())

    return input.__class__(
        columns=[
            aten.expand.default(
                column,
                size[:-1],
                implicit=implicit,
            )
            for column in input._columns
        ],
        size=tuple(
            old if new == -1 else new for old, new in zip(old_size, size)
        )[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.transpose.int)
def _transpose(
    input: ColumnarTensor,
    dim0: int,
    dim1: int,
) -> ColumnarTensor:

    dim0 = _normalize_dim(input, dim0)
    dim1 = _normalize_dim(input, dim1)

    if dim0 == dim1 == input.dim() - 1:
        return _alias(input)

    if input.dim() - 1 in (dim0, dim1):
        raise RuntimeError(
            f"Can't transpose the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    columns = [column.transpose(dim0, dim1) for column in input._columns]

    size = list(input.size())
    size[dim0], size[dim1] = size[dim1], size[dim0]

    return input.__class__(
        columns=columns,
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.permute.default)
def _permute(input: ColumnarTensor, dims: Sequence[int]) -> ColumnarTensor:
    dims = tuple(_normalize_dim(input, dim) for dim in dims)
    if dims[-1] != input.dim() - 1:
        raise RuntimeError(
            f"Can't permute the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=[column.permute(dims[:-1]) for column in input._columns],
        size=tuple(input.size(dim) for dim in dims[:-1]),
        device=input.device,
    )


@ColumnarTensor.implements(aten.select.int)
def _select(input: ColumnarTensor, dim: int, index: int) -> Tensor:
    dim = _normalize_dim(input, dim)

    if dim == input.dim() - 1:
        return aten.alias.default(input._columns[index])

    return input.__class__(
        columns=[column.select(dim, index) for column in input._columns],
        size=(*input.size()[:dim], *input.size()[dim + 1 : -1]),
        device=input.device,
    )


@ColumnarTensor.implements(aten.slice.Tensor)
def _slice(
    input: ColumnarTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> ColumnarTensor:
    dim = _normalize_dim(input, dim)

    if dim == input.dim() - 1:
        return input.__class__(
            columns=input._columns[slice(start, end, step)],
            size=input.size()[:-1],
            device=input.device,
        )

    return input.__class__(
        columns=[
            aten.slice.Tensor(column, dim, start, end, step)
            for column in input._columns
        ],
        size=(
            *input.size()[:dim],
            len(range(input.size(dim))[slice(start, end, step)]),
            *input.size()[dim + 1 : -1],
        ),
        device=input.device,
    )


@ColumnarTensor.implements(aten.narrow.default)
def _narrow(
    input: ColumnarTensor,
    dim: int,
    start: int,
    length: int,
) -> ColumnarTensor:
    return _slice(input, dim=dim, start=start, end=start + length)


@ColumnarTensor.implements(aten.unbind.int)
def _unbind(input: ColumnarTensor, dim: int = 0) -> tuple[Tensor, ...]:
    dim = _normalize_dim(input, dim)

    if dim == input.dim() - 1:
        return tuple(aten.alias.default(column) for column in input._columns)

    columns_list = [column.unbind(dim) for column in input._columns]
    if len(columns_list) == 0:
        size = (*input.size()[:dim], *input.size()[dim + 1 : -1])
        return tuple(
            input.__class__(
                columns=(),
                size=size,
                device=input.device,
            )
            for _ in range(input.size(dim))
        )

    return tuple(
        input.__class__(columns=columns, device=input.device)
        for columns in zip(*columns_list)
    )


@ColumnarTensor.implements(aten.split.Tensor)
def _split(
    input: ColumnarTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[ColumnarTensor, ...]:
    split_sizes = tuple(
        min(split_size, input.size(dim) - start)
        for start in range(0, input.size(dim), split_size)
    )
    return _split_with_sizes(input, split_sizes, dim=dim)


@ColumnarTensor.implements(aten.split.sizes)
@ColumnarTensor.implements(aten.split.default)
@ColumnarTensor.implements(aten.split_with_sizes.default)
def _split_with_sizes(
    input: ColumnarTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[ColumnarTensor, ...]:
    dim = _normalize_dim(input, dim)
    split_sizes = tuple(split_sizes)

    if dim == input.dim() - 1:
        if sum(split_sizes) != input.size(dim):
            raise RuntimeError(
                f"split_with_sizes expects split_sizes to sum exactly to "
                f"{input.size(dim)} (input tensor's size at dimension {dim}), "
                f"but got split_sizes={split_sizes}"
            )

        start = 0
        outs = []
        for split_size in split_sizes:
            out = input.__class__(
                columns=input._columns[start : start + split_size],
                size=input.size()[:-1],
                device=input.device,
            )
            start += split_size
            outs.append(out)
        return tuple(outs)

    columns_list = [col.split(split_sizes, dim=dim) for col in input._columns]
    if len(columns_list) == 0:
        return tuple(
            input.__class__(
                columns=(),
                size=(
                    *input.size()[:dim],
                    split_size,
                    *input.size()[dim + 1 : -1],
                ),
                device=input.device,
            )
            for split_size in split_sizes
        )

    return tuple(
        input.__class__(columns=columns, device=input.device)
        for columns in zip(*columns_list)
    )


@ColumnarTensor.implements(aten.index_select.default)
def _index_select(
    input: ColumnarTensor,
    dim: int,
    index: Tensor,
) -> ColumnarTensor:
    dim = _normalize_dim(input, dim)
    if dim == input.dim() - 1:
        return input.__class__(
            columns=[input._columns[i] for i in index.tolist()],
            size=input.size()[:-1],
            device=input.device,
        )

    return input.__class__(
        columns=[column.index_select(dim, index) for column in input._columns],
        size=(
            *input.size()[:dim],
            index.numel(),
            *input.size()[dim + 1 : -1],
        ),
        device=input.device,
    )


@ColumnarTensor.implements(aten.index.Tensor)
def _index(
    input: ColumnarTensor,
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
        if current_dim <= input.dim() - 1 < current_dim + num_indexed_dims:
            if num_indexed_dims != 1:
                raise RuntimeError(
                    f"Can't index the column dimension of "
                    f"'{input.__class__.__name__}' with a multi-dimensional "
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
                f"'{input.__class__.__name__}' together with other dimensions"
            )
        if column_index.dtype == torch.bool:
            column_index = column_index.nonzero().view(-1)
        return input.__class__(
            columns=[input._columns[i] for i in column_index.tolist()],
            size=input.size()[:-1],
            device=input.device,
        )

    if len(input._columns) == 0:
        dummy = torch.empty(input.size(), device=input.device)
        return input.__class__(
            columns=(),
            size=aten.index.Tensor(dummy, indices).size()[:-1],
            device=input.device,
        )

    return input.__class__(
        columns=[
            aten.index.Tensor(column, indices) for column in input._columns
        ],
        device=input.device,
    )


@ColumnarTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> ColumnarTensor:
    if not all(isinstance(tensor, ColumnarTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be '{ColumnarTensor.__name__}' instances"
        )

    tensors = cast(Sequence[ColumnarTensor], tensors)
    dim = _normalize_dim(tensors[0], dim)

    if dim == tensors[0].dim() - 1:
        return tensors[0].__class__(
            columns=tuple(chain.from_iterable(t._columns for t in tensors)),
            size=tensors[0].size()[:-1],
            device=tensors[0].device,
        )

    return tensors[0].__class__(
        columns=[
            torch.cat([tensor._columns[i] for tensor in tensors], dim=dim)
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
            f"Expected all tensors to be '{ColumnarTensor.__name__}' instances"
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
            f"'{tensors[0].__class__.__name__}'"
        )

    return tensors[0].__class__(
        columns=[
            torch.stack([tensor._columns[i] for tensor in tensors], dim=dim)
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


def _normalize_dim(input: Tensor, dim: int) -> int:
    if dim < -input.dim() or dim >= input.dim():
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.dim()}, {input.dim() - 1}], but got {dim})"
        )
    return dim % input.dim()
