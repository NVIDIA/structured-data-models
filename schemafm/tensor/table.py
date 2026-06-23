from collections.abc import Callable, Iterator, Mapping, Sequence
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
def _is_pinned(input: TableTensor, device: torch.device | None = None) -> bool:
    return all(tensor.is_pinned(device) for _, tensor in input.items())


@TableTensor.implements(aten._pin_memory.default)
def _pin_memory(
    input: TableTensor,
    device: torch.device | None = None,
) -> TableTensor:
    blocks = {stype: tensor.pin_memory() for stype, tensor in input.items()}
    return input.__class__(
        columns=cast(dict[StypeLike, tuple[str, ...]], input._columns),
        **blocks,
    )


# Helpers #####################################################################


def _block_size_repr(size: Sequence[int]) -> str:
    if len(size) == 0:
        return "(*,)"
    if len(size) == 1:
        return f"({size[0]}, *)"
    return f"{str(tuple(size))[:-1]}, *)"
