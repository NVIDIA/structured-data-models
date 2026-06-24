import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import chain
from typing import Any, ClassVar, SupportsIndex, TypeVar, cast

import torch
from torch import Tensor

from schemafm import Stype, StypeLike
from schemafm.tensor import CategoricalTensor

aten = torch.ops.aten

SelfTableTensor = TypeVar("SelfTableTensor", bound="TableTensor")


class TableTensor(Tensor):
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _numerical: Tensor
    _categorical: CategoricalTensor
    _columns: dict[Stype, tuple[str, ...]]
    _column_to_loc: dict[str, tuple[Stype, int]]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        self,
        size: Sequence[int] | None = None,
        columns: Mapping[StypeLike, Sequence[str]] | None = None,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        pass

    def __new__(
        cls: type[SelfTableTensor],
        size: Sequence[int] | None = None,
        columns: Mapping[StypeLike, Sequence[str]] | None = None,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        device: torch.device | str | None = None,
    ) -> SelfTableTensor:

        if size is not None:
            size = tuple(size)
            if len(size) == 0:
                raise ValueError("Expected 'size' to be non-empty")

        size = tuple(size) if size is not None else size
        device = torch.device(device) if device is not None else device

        for stype, block in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
        ):
            if block is None:
                continue

            size = tuple(block.size()[:-1]) if size is None else size
            device = block.device if device is None else device

            if block.dim() < 2:
                raise ValueError(
                    f"Expected '{stype.value}' block to be at least 2D "
                    f"(got {block.dim()}D)"
                )
            if size != block.size()[:-1]:
                raise ValueError(
                    f"Expected '{stype.value}' block size of "
                    f"{_block_size_repr(size)} (got {tuple(block.size())})"
                )
            if device != block.device:
                raise ValueError(
                    f"Expected '{stype.value}' block to be on device "
                    f"'{device}' (got '{block.device}')"
                )

        if size is None:
            raise ValueError(
                "Expected 'size' to be given when all blocks are 'None'"
            )

        if numerical is None:
            numerical = torch.empty((*size, 0), device=device)
        if categorical is None:
            categorical = CategoricalTensor(
                data=torch.empty((*size, 0), dtype=torch.int32, device=device),
                categories=(),
            )

        columns = {
            Stype(stype): tuple(names)
            for stype, names in (columns or {}).items()
        }
        columns = {
            Stype.numerical: tuple(columns.get(Stype.numerical, ())),
            Stype.categorical: tuple(columns.get(Stype.categorical, ())),
        }

        for stype, block in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
        ):
            if block.size(-1) != len(columns[stype]):
                _columns = "column" if len(columns[stype]) == 1 else "columns"
                raise ValueError(
                    f"Expected '{stype.value}' block to hold "
                    f"{len(columns[stype])} {_columns} (got {block.size(-1)})"
                )

        column_names = list(chain.from_iterable(columns.values()))
        column_to_loc: dict[str, tuple[Stype, int]] = {}
        for stype, names in columns.items():
            for i, name in enumerate(names):
                column_to_loc[name] = (Stype(stype), i)
        if len(column_names) != len(column_to_loc):
            raise ValueError("Expected column names to be unique")

        out = Tensor._make_wrapper_subclass(
            cls,
            size=(*size, len(column_names)),
            dtype=numerical.dtype,
            device=numerical.device,
            requires_grad=False,
        )

        out._numerical = numerical
        out._categorical = categorical
        out._columns = columns
        out._column_to_loc = column_to_loc

        return out

    # Properties ##############################################################

    @property
    def columns(self) -> Mapping[Stype, tuple[str, ...]]:
        return self._columns.copy()

    @property
    def numerical(self) -> Tensor:
        return self._numerical

    @property
    def categorical(self) -> CategoricalTensor:
        return self._categorical

    def items(self) -> Iterator[tuple[Stype, Tensor]]:
        yield Stype.numerical, self._numerical
        yield Stype.categorical, self._categorical

    @property
    def blocks(self) -> Mapping[Stype, Tensor]:
        return dict(self.items())

    def select_columns(self, columns: str | Iterable[str]) -> "TableTensor":
        columns = {columns} if isinstance(columns, str) else set(columns)

        for column in columns:
            if column not in self._column_to_loc:
                raise KeyError(column)

        index_dict: dict[Stype, list[int]] = defaultdict(list)
        columns_dict: dict[StypeLike, list[str]] = defaultdict(list)
        for stype, stype_columns in self._columns.items():
            for i, column in enumerate(stype_columns):
                if column in columns:
                    index_dict[stype].append(i)
                    columns_dict[stype].append(column)

        blocks: dict[Stype, Tensor] = {}
        for stype, tensor in self.items():
            indices = index_dict[stype]
            if len(indices) == 0:
                blocks[stype] = tensor.narrow(-1, 0, 0)
            elif len(indices) == len(self._columns[stype]):
                blocks[stype] = tensor
            else:
                index = torch.tensor(indices, device=tensor.device)
                blocks[stype] = tensor.index_select(-1, index)

        return self.__class__(columns=columns_dict, **blocks)

    def drop_columns(self, columns: str | Iterable[str]) -> "TableTensor":
        columns = {columns} if isinstance(columns, str) else set(columns)

        for column in columns:
            if column not in self._column_to_loc:
                raise KeyError(column)

        columns = set(chain.from_iterable(self._columns.values())) - columns
        return self.select_columns(columns)

    # Decorators ##############################################################

    @classmethod
    def implements(
        cls,
        torch_function: Callable[..., Any],
    ) -> Callable[..., Any]:
        if "HANDLED_FUNCTIONS" not in cls.__dict__:
            cls.HANDLED_FUNCTIONS = cls.HANDLED_FUNCTIONS.copy()

        def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
            cls.HANDLED_FUNCTIONS[torch_function] = my_function
            return my_function

        return decorator

    # PyTorch/Python builtins #################################################

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        args = (
            tuple(self.size()[:-1]),
            self._columns,
            self._numerical,
            self._categorical,
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
            f"'{func}' is not supported for '{cls.__name__}'"
        )

    def __getitem__(self, indices: Any) -> "TableTensor":
        def is_column_index(index: Any) -> bool:
            return isinstance(index, str) or (
                isinstance(index, list)
                and all(isinstance(value, str) for value in index)
            )

        if is_column_index(indices):
            return self.select_columns(indices)
        if not isinstance(indices, tuple):
            return cast(TableTensor, Tensor.__getitem__(self, indices))
        if not any(is_column_index(index) for index in indices):
            return cast(TableTensor, Tensor.__getitem__(self, indices))
        if any(is_column_index(index) for index in indices[:-1]):
            raise IndexError(
                "Column names can only index the column dimension"
            )

        out = Tensor.__getitem__(self, (*indices[:-1], slice(None)))
        return cast(TableTensor, out).select_columns(indices[-1])

    def is_shared(self) -> bool:
        return all(tensor.is_shared() for _, tensor in self.items())

    def share_memory_(self) -> "TableTensor":
        for _, tensor in self.items():
            tensor.share_memory_()
        return self

    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            tensor.is_contiguous(memory_format=memory_format)
            for _, tensor in self.items()
        )

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> "TableTensor":
        if self.is_contiguous(memory_format=memory_format):
            return self
        return _contiguous(self, memory_format=memory_format)

    def tolist() -> Any:
        raise NotImplementedError("'tolist() is not yet implemented")  # TODO

    def __repr__(self, *, tensor_contents: Any = None) -> str:
        def _columns_repr(
            columns: Sequence[str],
            max_cols: int = 3,
            max_item_len: int = 24,
        ) -> str:
            columns = [
                f"'{column}'"
                if len(column) <= max_item_len
                else column[: max_item_len - 1] + "…"
                for column in columns
            ]
            if len(columns) > max_cols:
                [*columns[: max_cols - 1], "...", columns[-1]]
            return "[" + ", ".join(column for column in columns) + "]"

        stype_repr = [
            (
                f"    {stype.value} ({tensor.size(-1):,}): "
                f"{_columns_repr(self._columns[stype])},"
            )
            for stype, tensor in self.items()
        ]

        out = f"{self.__class__.__name__}(\n"
        out += f"  size={tuple(self.size())},\n"
        out += "  blocks={\n"
        out += "\n".join(stype_repr) + "\n"
        out += "  },\n"
        if self.device.type != "cpu":
            out += f"  device={self.device},\n"
        out += ")"
        return out


@TableTensor.implements(aten.alias.default)
def _alias(input: TableTensor) -> TableTensor:
    blocks = {
        stype: aten.alias.default(tensor) for stype, tensor in input.items()
    }
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten._to_copy.default)
def _to_copy(
    input: TableTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | str | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    blocks = {
        stype: aten._to_copy.default(
            tensor,
            device=device,
            dtype=dtype
            if not isinstance(tensor, CategoricalTensor)
            or dtype in (torch.int32, torch.int64)
            else None,
            layout=layout,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )
        for stype, tensor in input.items()
    }
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.clone.default)
def _clone(
    input: TableTensor,
    *,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    return _to_copy(input, memory_format=memory_format)


@TableTensor.implements(aten.contiguous.default)
def _contiguous(
    input: TableTensor,
    *,
    memory_format: torch.memory_format = torch.contiguous_format,
) -> TableTensor:
    blocks = {
        stype: tensor.contiguous(memory_format=memory_format)
        for stype, tensor in input.items()
    }
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.is_pinned.default)
def _is_pinned(input: TableTensor) -> bool:
    return all(tensor.is_pinned() for _, tensor in input.items())


@TableTensor.implements(aten._pin_memory.default)
def _pin_memory(input: TableTensor) -> TableTensor:
    blocks = {stype: tensor.pin_memory() for stype, tensor in input.items()}
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.view.default)
def _view(input: TableTensor, size: Sequence[int]) -> TableTensor:
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

    blocks = {
        stype: tensor.view((*size[:-1], tensor.size(-1)))
        for stype, tensor in input.items()
    }
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten._unsafe_view.default)
def _unsafe_view(input: TableTensor, size: Sequence[int]) -> TableTensor:
    return _view(input, size)


@TableTensor.implements(aten.squeeze.default)
def _squeeze(input: TableTensor) -> TableTensor:
    return _squeeze_dims(input, range(input.dim() - 1))


@TableTensor.implements(aten.squeeze.dim)
def _squeeze_dim(input: TableTensor, dim: int) -> TableTensor:
    blocks = {stype: tensor.squeeze(dim) for stype, tensor in input.items()}

    if dim % input.dim() == input.dim() - 1:
        raise RuntimeError(
            f"Can't squeeze the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.squeeze.dims)
def _squeeze_dims(input: TableTensor, dim: Sequence[int]) -> TableTensor:
    dims = tuple(dim)
    blocks = {stype: tensor.squeeze(dims) for stype, tensor in input.items()}

    if input.dim() - 1 in tuple(dim % input.dim() for dim in dims):
        raise RuntimeError(
            f"Can't squeeze the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.unsqueeze.default)
def _unsqueeze(input: TableTensor, dim: int) -> TableTensor:
    blocks = {stype: tensor.unsqueeze(dim) for stype, tensor in input.items()}

    if dim % (input.dim() + 1) == input.dim():
        raise RuntimeError(
            f"Can't unsqueeze after the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.expand.default)
def _expand(
    input: TableTensor,
    size: Sequence[int],
    *,
    implicit: bool = False,
) -> TableTensor:
    size = tuple(size)
    if len(size) == 0 or size[-1] not in (-1, input.size(-1)):
        _columns = "column" if input.size(-1) == 1 else "columns"
        raise RuntimeError(
            f"Can't expand '{input.__class__.__name__}' with "
            f"{input.size(-1)} {_columns} to shape {size}"
        )

    blocks = {
        stype: aten.expand.default(
            tensor,
            (*size[:-1], tensor.size(-1)),
            implicit=implicit,
        )
        for stype, tensor in input.items()
    }
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.transpose.int)
def _transpose(input: TableTensor, dim0: int, dim1: int) -> TableTensor:
    blocks = {
        stype: tensor.transpose(dim0, dim1) for stype, tensor in input.items()
    }

    dim0 %= input.dim()
    dim1 %= input.dim()
    if dim0 != dim1 and input.dim() - 1 in (dim0, dim1):
        raise RuntimeError(
            f"Can't transpose the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.permute.default)
def _permute(input: TableTensor, dims: Sequence[int]) -> TableTensor:
    dims = tuple(dims)
    blocks = {stype: tensor.permute(dims) for stype, tensor in input.items()}

    dims = tuple(dim % input.dim() for dim in dims)
    if dims[-1] != input.dim() - 1:
        raise RuntimeError(
            f"Can't permute the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.select.int)
def _select(input: TableTensor, dim: int, index: int) -> TableTensor:
    if _is_column_dim(input, dim):
        raise RuntimeError(
            f"Can't select the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    blocks = {
        stype: tensor.select(dim, index) for stype, tensor in input.items()
    }

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.slice.Tensor)
def _slice(
    input: TableTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> TableTensor:
    blocks = {
        stype: aten.slice.Tensor(tensor, dim, start, end, step)
        for stype, tensor in input.items()
    }

    if dim % input.dim() == input.dim() - 1:
        raise RuntimeError(
            f"Can't slice the column dimension of '{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.narrow.default)
def _narrow(
    input: TableTensor,
    dim: int,
    start: int,
    length: int,
) -> TableTensor:
    blocks = {
        stype: tensor.narrow(dim, start, length)
        for stype, tensor in input.items()
    }

    if dim % input.dim() == input.dim() - 1:
        raise RuntimeError(
            f"Can't narrow the column dimension of "
            f"'{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.unbind.int)
def _unbind(input: TableTensor, dim: int = 0) -> tuple[TableTensor, ...]:
    if _is_column_dim(input, dim):
        return _split(input, split_size=1, dim=dim)

    tensors_dict: dict[Stype, tuple[Tensor, ...]] = {
        stype: tensor.unbind(dim) for stype, tensor in input.items()
    }

    stypes = tuple(tensors_dict.keys())
    return tuple(
        input.__class__(
            columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
            **dict(zip(stypes, blocks)),
        )
        for blocks in zip(*tensors_dict.values())
    )


@TableTensor.implements(aten.split.Tensor)
def _split(
    input: TableTensor,
    split_size: int,
    dim: int = 0,
) -> tuple[TableTensor, ...]:
    tensors_dict: dict[Stype, tuple[Tensor, ...]] = {
        stype: tensor.split(split_size, dim) for stype, tensor in input.items()
    }

    if dim % input.dim() == input.dim() - 1:
        if split_size != 1:
            raise RuntimeError(
                f"Can only split the column dimension of "
                f"'{input.__class__.__name__}' with split size 1"
            )

        return tuple(
            input.__class__(
                columns={stype: (input._columns[stype][i],)},
                **{stype: tensor},
            )
            for stype, tensors in tensors_dict.items()
            for i, tensor in enumerate(tensors)
        )

    stypes = tuple(tensors_dict.keys())
    return tuple(
        input.__class__(
            columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
            **dict(zip(stypes, blocks)),
        )
        for blocks in zip(*tensors_dict.values())
    )


@TableTensor.implements(aten.split.sizes)
@TableTensor.implements(aten.split.default)
@TableTensor.implements(aten.split_with_sizes.default)
def _split_with_sizes(
    input: TableTensor,
    split_sizes: Sequence[int],
    dim: int = 0,
) -> tuple[TableTensor, ...]:
    if _is_column_dim(input, dim):
        raise RuntimeError(
            f"Can't split the column dimension of '{input.__class__.__name__}'"
        )

    split_sizes = tuple(split_sizes)
    blocks_dict: dict[Stype, tuple[Tensor, ...]] = {
        stype: tensor.split(split_sizes, dim)
        for stype, tensor in input.items()
    }

    stypes = tuple(blocks_dict.keys())
    return tuple(
        input.__class__(
            columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
            **dict(zip(stypes, blocks)),
        )
        for blocks in zip(*blocks_dict.values())
    )


@TableTensor.implements(aten.index_select.default)
def _index_select(
    input: TableTensor,
    dim: int,
    index: Tensor,
) -> TableTensor:
    blocks = {
        stype: tensor.index_select(dim, index)
        for stype, tensor in input.items()
    }

    if dim % input.dim() == input.dim() - 1:
        raise RuntimeError(
            f"Can't index the column dimension of '{input.__class__.__name__}'"
        )

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.index.Tensor)
def _index(
    input: TableTensor,
    indices: Sequence[Tensor | None],
) -> TableTensor:

    current_dim = 0
    for index in indices:
        if index is None:
            current_dim += 1
            continue

        # Check whether we index the column dimension:
        num_indexed_dims = index.dim() if index.dtype == torch.bool else 1
        if current_dim <= input.dim() - 1 < current_dim + num_indexed_dims:
            raise RuntimeError(
                f"Can't index the column dimension of "
                f"'{input.__class__.__name__}'"
            )
        current_dim += num_indexed_dims

    blocks = {
        stype: aten.index.Tensor(tensor, indices)
        for stype, tensor in input.items()
    }

    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


@TableTensor.implements(aten.cat.default)
def _cat(tensors: Sequence[Tensor], dim: int = 0) -> TableTensor:
    if not all(isinstance(tensor, TableTensor) for tensor in tensors):
        raise TypeError(
            f"Expected all tensors to be '{TableTensor.__name__}' instances"
        )

    tensors = cast(Sequence[TableTensor], tensors)
    blocks = {
        stype: torch.cat([tensor.blocks[stype] for tensor in tensors], dim=dim)
        for stype, _ in tensors[0].items()
    }

    if dim % tensors[0].dim() != tensors[0].dim() - 1:
        columns = tensors[0]._columns
    else:
        columns = {
            stype: tuple(
                chain.from_iterable(t._columns[stype] for t in tensors)
            )
            for stype, _ in tensors[0].items()
        }
    return tensors[0].__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], columns),
        **blocks,
    )


@TableTensor.implements(aten.stack.default)
def _stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not all(isinstance(tensor, TableTensor) for tensor in tensors):
        raise TypeError("Expected all tensors to be 'TableTensor' instances")

    tensors = cast(Sequence[TableTensor], tensors)
    blocks = {
        stype: torch.stack([tensor.blocks[stype] for tensor in tensors], dim)
        for stype in tensors[0]._columns
    }

    dim %= tensors[0].dim() + 1
    if dim >= tensors[0].dim():
        raise RuntimeError(
            f"Can't stack after the column dimension of "
            f"'{tensors[0].__class__.__name__}'"
        )

    return tensors[0].__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], tensors[0]._columns),
        **blocks,
    )


# Helpers #####################################################################


def _block_size_repr(size: Sequence[int]) -> str:
    if len(size) == 0:
        return "(*,)"
    if len(size) == 1:
        return f"({size[0]}, *)"
    return f"{str(tuple(size))[:-1]}, *)"


def _is_column_dim(input: TableTensor, dim: int) -> bool:
    if dim < -input.dim() or dim >= input.dim():
        return False
    return dim % input.dim() == input.dim() - 1
