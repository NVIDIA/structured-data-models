from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from typing import Any, ClassVar, TypeVar

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
            dtype=torch.uint8,  # NOTE Dummy. DO NOT USE.
            device=numerical.device,
            requires_grad=False,
        )

        out._numerical = numerical
        out._categorical = categorical
        out._columns = columns
        out._column_to_loc = column_to_loc

        return out

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

    @property
    def dtype(self) -> torch.dtype:
        raise RuntimeError(
            f"'{self.__class__.__name__}' does not have a single dtype"
        )


# Helpers #####################################################################


def _block_size_repr(size: Sequence[int]) -> str:
    if len(size) == 0:
        return "(*,)"
    if len(size) == 1:
        return f"({size[0]}, *)"
    return f"{str(tuple(size))[:-1]}, *)"
