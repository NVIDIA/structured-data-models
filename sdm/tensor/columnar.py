import math
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, ClassVar, SupportsIndex, TypeVar, cast

import pyarrow as pa
import torch
from torch import Tensor
from typing_extensions import override

from sdm.tensor import CategoricalTensor, StringTensor

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
        columns = tuple(columns)
        device = torch.device(device) if device is not None else None

        if size is None:
            if len(columns) == 0:
                raise ValueError(
                    "Expected 'size' to be given for zero columnar data"
                )
            if isinstance(columns[0], CategoricalTensor):
                size = tuple(columns[0].size()[:-1])
            else:
                size = tuple(columns[0].size())
        else:
            size = tuple(size)

        if len(size) < 1:
            raise ValueError(
                f"Expected 'size' to have at least one dimension "
                f"(got {len(size)})"
            )

        for i, column in enumerate(columns):
            if isinstance(column, CategoricalTensor):
                if column.size(-1) != 1:
                    raise ValueError(
                        f"Expected categorical tensor {i} in '{cls.__name__}' "
                        f"to only hold a single column"
                    )
                column_size = column.size()[:-1]
            else:
                column_size = column.size()
            if column_size != size:
                raise ValueError(
                    f"Expected value {i} in '{cls.__name__}' to have size "
                    f"{size} (got {tuple(column_size)})"
                )

            device = column.device if device is None else device
            if column.device != device:
                raise ValueError(
                    f"Expected value {i} in '{cls.__name__}' to be on "
                    f"device '{device}' (got '{column.device}')"
                )

        if device is None:
            device = torch.get_default_device()

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
        if self.device.type != "cpu":
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
    size = _infer_view_size(input, size)
    return input.__class__(
        columns=[
            column.view(_column_size(column, size[:-1]))
            for column in input._columns
        ],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten._unsafe_view.default)
def _unsafe_view(
    input: ColumnarTensor,
    size: Sequence[int],
) -> ColumnarTensor:
    size = _infer_view_size(input, size)
    return input.__class__(
        columns=[
            aten._unsafe_view(column, _column_size(column, size[:-1]))
            for column in input._columns
        ],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.squeeze.default)
def _squeeze(input: ColumnarTensor) -> ColumnarTensor:
    return _squeeze_dims(input, range(input.dim() - 1))


@ColumnarTensor.implements(aten.squeeze.dim)
def _squeeze_dim(input: ColumnarTensor, dim: int) -> ColumnarTensor:
    return _squeeze_dims(input, (dim,))


@ColumnarTensor.implements(aten.squeeze.dims)
def _squeeze_dims(
    input: ColumnarTensor,
    dim: Sequence[int],
) -> ColumnarTensor:
    dims = tuple(d % input.dim() for d in dim)
    _raise_if_column_dim(input, dims)
    return input.__class__(
        columns=[column.squeeze(dims) for column in input._columns],
        size=_squeeze_size(input.size(), dims)[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.unsqueeze.default)
def _unsqueeze(input: ColumnarTensor, dim: int) -> ColumnarTensor:
    dim %= input.dim() + 1
    if dim == input.dim():
        raise RuntimeError(
            f"Can't unsqueeze after the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    size = (*input.size()[:dim], 1, *input.size()[dim:])
    return input.__class__(
        columns=[column.unsqueeze(dim) for column in input._columns],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.expand.default)
def _expand(
    input: ColumnarTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> ColumnarTensor:
    size = _expand_size(input, size)
    return input.__class__(
        columns=[
            aten.expand.default(
                column,
                _column_size(column, size[:-1]),
                implicit=implicit,
            )
            for column in input._columns
        ],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.transpose.int)
def _transpose(
    input: ColumnarTensor,
    dim0: int,
    dim1: int,
) -> ColumnarTensor:
    dim0 %= input.dim()
    dim1 %= input.dim()
    _raise_if_column_dim(input, (dim0, dim1))

    size = list(input.size())
    size[dim0], size[dim1] = size[dim1], size[dim0]
    return input.__class__(
        columns=[column.transpose(dim0, dim1) for column in input._columns],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.permute.default)
def _permute(input: ColumnarTensor, dims: Sequence[int]) -> ColumnarTensor:
    dims = tuple(dim % input.dim() for dim in dims)
    if dims[-1] != input.dim() - 1:
        raise RuntimeError(
            f"Can't permute the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    size = tuple(input.size(dim) for dim in dims)
    return input.__class__(
        columns=[
            column.permute(dims)
            if isinstance(column, CategoricalTensor)
            else column.permute(dims[:-1])
            for column in input._columns
        ],
        size=size[:-1],
        device=input.device,
    )


@ColumnarTensor.implements(aten.select.int)
def _select(input: ColumnarTensor, dim: int, index: int) -> Tensor:
    dim %= input.dim()
    if _is_column_dim(input, dim):
        return aten.alias.default(input._columns[index])

    size = (*input.size()[:dim], *input.size()[dim + 1 : -1])
    return input.__class__(
        columns=[column.select(dim, index) for column in input._columns],
        size=size,
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
    dim %= input.dim()
    if _is_column_dim(input, dim):
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
        size=_slice_size(input.size()[:-1], dim, start, end, step),
        device=input.device,
    )


@ColumnarTensor.implements(aten.narrow.default)
def _narrow(
    input: ColumnarTensor,
    dim: int,
    start: int,
    length: int,
) -> ColumnarTensor:
    dim %= input.dim()
    if _is_column_dim(input, dim):
        return input.__class__(
            columns=input._columns[start : start + length],
            size=input.size()[:-1],
            device=input.device,
        )

    size = list(input.size()[:-1])
    size[dim] = length
    return input.__class__(
        columns=[
            column.narrow(dim=dim, start=start, length=length)
            for column in input._columns
        ],
        size=size,
        device=input.device,
    )


@ColumnarTensor.implements(aten.unbind.int)
def _unbind(input: ColumnarTensor, dim: int = 0) -> tuple[Tensor, ...]:
    dim %= input.dim()
    if _is_column_dim(input, dim):
        return tuple(aten.alias.default(column) for column in input._columns)

    columns_list = [column.unbind(dim) for column in input._columns]
    if len(columns_list) == 0:
        size = (*input.size()[:dim], *input.size()[dim + 1 : -1])
        return tuple(
            input.__class__(columns=(), size=size, device=input.device)
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
    dim %= input.dim()
    if _is_column_dim(input, dim):
        return tuple(
            input.__class__(
                columns=input._columns[i : i + split_size],
                size=input.size()[:-1],
                device=input.device,
            )
            for i in range(0, input.size(-1), split_size)
        )

    columns_list = [
        column.split(split_size, dim=dim) for column in input._columns
    ]
    return _wrap_split(input, columns_list, dim=dim, split_size=split_size)


@ColumnarTensor.implements(aten.split.sizes)
@ColumnarTensor.implements(aten.split.default)
@ColumnarTensor.implements(aten.split_with_sizes.default)
def _split_with_sizes(
    input: ColumnarTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[ColumnarTensor, ...]:
    dim %= input.dim()
    split_sizes = tuple(split_sizes)
    if _is_column_dim(input, dim):
        end = 0
        out = []
        for split_size in split_sizes:
            start, end = end, end + split_size
            out.append(
                input.__class__(
                    columns=input._columns[start:end],
                    size=input.size()[:-1],
                    device=input.device,
                )
            )
        return tuple(out)

    columns_list = [
        column.split(split_sizes, dim=dim) for column in input._columns
    ]
    return _wrap_split(input, columns_list, dim=dim, split_sizes=split_sizes)


@ColumnarTensor.implements(aten.index_select.default)
def _index_select(
    input: ColumnarTensor,
    dim: int,
    index: Tensor,
) -> ColumnarTensor:
    dim %= input.dim()
    if _is_column_dim(input, dim):
        return input.__class__(
            columns=[
                aten.alias.default(input._columns[i]) for i in index.tolist()
            ],
            size=input.size()[:-1],
            device=input.device,
        )

    size = list(input.size()[:-1])
    size[dim] = index.numel()
    return input.__class__(
        columns=[column.index_select(dim, index) for column in input._columns],
        size=size,
        device=input.device,
    )


@ColumnarTensor.implements(aten.index.Tensor)
def _index(
    input: ColumnarTensor,
    indices: Sequence[Tensor | None],
) -> ColumnarTensor:
    current_dim = 0
    column_index: Tensor | None = None
    has_other_index = False
    for index in indices:
        if index is None:
            current_dim += 1
            continue

        num_indexed_dims = index.dim() if index.dtype == torch.bool else 1
        if current_dim <= input.dim() - 1 < current_dim + num_indexed_dims:
            if num_indexed_dims != 1:
                raise RuntimeError(
                    f"Can't index the column dimension of "
                    f"'{input.__class__.__name__}'"
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
            columns=[
                aten.alias.default(input._columns[i])
                for i in column_index.tolist()
            ],
            size=input.size()[:-1],
            device=input.device,
        )

    if len(input._columns) == 0:
        dummy = torch.empty(input.size()[:-1], device=input.device)
        return input.__class__(
            columns=(),
            size=aten.index.Tensor(dummy, indices).size(),
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
    dim %= tensors[0].dim()
    if _is_column_dim(tensors[0], dim):
        return tensors[0].__class__(
            columns=tuple(chain.from_iterable(t._columns for t in tensors)),
            size=tensors[0].size()[:-1],
            device=tensors[0].device,
        )

    size = list(tensors[0].size()[:-1])
    size[dim] = sum(tensor.size(dim) for tensor in tensors)
    return tensors[0].__class__(
        columns=[
            torch.cat([tensor._columns[i] for tensor in tensors], dim=dim)
            for i in range(tensors[0].size(-1))
        ],
        size=size,
        device=tensors[0].device,
    )


@ColumnarTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> ColumnarTensor:
    if not all(isinstance(tensor, ColumnarTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be '{ColumnarTensor.__name__}' instances"
        )

    tensors = cast(Sequence[ColumnarTensor], tensors)
    dim %= tensors[0].dim() + 1
    if dim >= tensors[0].dim():
        raise RuntimeError(
            f"Can't stack after the column dimension of "
            f"'{tensors[0].__class__.__name__}'"
        )

    size = list(tensors[0].size()[:-1])
    size.insert(dim, len(tensors))
    return tensors[0].__class__(
        columns=[
            torch.stack([tensor._columns[i] for tensor in tensors], dim=dim)
            for i in range(tensors[0].size(-1))
        ],
        size=size,
        device=tensors[0].device,
    )


# Helpers #####################################################################


def _column_size(column: Tensor, size: Sequence[int]) -> tuple[int, ...]:
    size = tuple(size)
    if isinstance(column, CategoricalTensor):
        return (*size, 1)
    return size


def _infer_view_size(
    input: ColumnarTensor,
    size: Sequence[int],
) -> tuple[int, ...]:
    size = tuple(size)
    if size.count(-1) > 1:
        raise RuntimeError("Only one dimension can be inferred")

    if -1 in size:
        known = math.prod(dim_size for dim_size in size if dim_size != -1)
        if known == 0 or input.numel() % known != 0:
            raise RuntimeError(
                f"Shape {size} is invalid for input of size {input.numel()}"
            )
        dim = size.index(-1)
        size = (*size[:dim], input.numel() // known, *size[dim + 1 :])

    if len(size) == 0 or size[-1] != input.size(-1):
        raise RuntimeError(
            f"Can't reshape '{input.__class__.__name__}' with "
            f"{input.size(-1)} columns into shape {size}"
        )

    return size


def _expand_size(
    input: ColumnarTensor,
    size: Sequence[int],
) -> tuple[int, ...]:
    size = tuple(size)
    if len(size) == 0 or size[-1] not in (-1, input.size(-1)):
        raise RuntimeError(
            f"Can't expand '{input.__class__.__name__}' with "
            f"{input.size(-1)} columns to shape {size}"
        )
    return tuple(
        input.size(i) if dim_size == -1 else dim_size
        for i, dim_size in enumerate(size)
    )


def _squeeze_size(
    size: Sequence[int],
    dims: Sequence[int],
) -> tuple[int, ...]:
    dims_set = set(dims)
    return tuple(
        dim_size
        for i, dim_size in enumerate(size)
        if i not in dims_set or dim_size != 1
    )


def _raise_if_column_dim(
    input: ColumnarTensor,
    dims: Sequence[int],
) -> None:
    if input.dim() - 1 in dims:
        raise RuntimeError(
            f"Can't operate on the column dimension of "
            f"'{input.__class__.__name__}'"
        )


def _is_column_dim(input: ColumnarTensor, dim: int) -> bool:
    return dim % input.dim() == input.dim() - 1


def _slice_size(
    size: Sequence[int],
    dim: int,
    start: int | None,
    end: int | None,
    step: int,
) -> tuple[int, ...]:
    out = list(size)
    out[dim] = len(range(size[dim])[slice(start, end, step)])
    return tuple(out)


def _wrap_split(
    input: ColumnarTensor,
    columns_list: Sequence[Sequence[Tensor]],
    *,
    dim: int,
    split_size: int | None = None,
    split_sizes: Sequence[int] | None = None,
) -> tuple[ColumnarTensor, ...]:
    if len(columns_list) == 0:
        if split_sizes is None:
            assert split_size is not None
            split_sizes = tuple(
                min(split_size, input.size(dim) - i)
                for i in range(0, input.size(dim), split_size)
            )

        return tuple(
            input.__class__(
                columns=(),
                size=(*input.size()[:dim], size, *input.size()[dim + 1 : -1]),
                device=input.device,
            )
            for size in split_sizes
        )

    return tuple(
        input.__class__(columns=columns, device=input.device)
        for columns in zip(*columns_list)
    )
