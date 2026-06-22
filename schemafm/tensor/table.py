from collections.abc import Callable, Mapping, Sequence
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
        columns: Mapping[StypeLike, Sequence[str]] | None = None,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        pass

    def __new__(
        cls: type[SelfTableTensor],
        columns: Mapping[StypeLike, Sequence[str]] | None = None,
        numerical: Tensor | None = None,
        categorical: CategoricalTensor | None = None,
        size: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> SelfTableTensor:

        size = tuple(size) if size is not None else size
        device = torch.device(device) if device is not None else device

        for block in (numerical, categorical):
            if block is None:
                continue

            size = tuple(block.size()[:-1]) if size is None else size
            device = block.device if device is None else device

            if size != block.size()[:-1]:
                raise ValueError(
                    f"Expected block size of '{size}' "
                    f"(got '{tuple(block.size()[:-1])}')"
                )
            if device != block.device:
                raise ValueError(
                    f"Expected block to be on device '{device}' "
                    f"(got '{block.device}')"
                )

        if size is None:
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}' to be given when "
                f"all blocks are 'None'"
            )
        elif len(size) < 1:
            raise ValueError("Expected table to hold at least two dimensions")

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

        if numerical.size(-1) != len(columns[Stype.numerical]):
            raise ValueError(
                f"Expected 'numerical' block in '{cls.__name__}' to hold "
                f"{len(columns[Stype.numerical])} columns "
                f"(got {numerical.size(-1)})"
            )
        if categorical.size(-1) != len(columns[Stype.categorical]):
            raise ValueError(
                f"Expected 'categorical' block in '{cls.__name__}' to hold "
                f"{len(columns[Stype.categorical])} columns "
                f"(got {categorical.size(-1)})"
            )

        num_columns = sum(len(names) for names in columns.values())
        column_to_loc: dict[str, tuple[Stype, int]] = {}
        for stype, names in columns.items():
            for i, name in enumerate(names):
                column_to_loc[name] = (Stype(stype), i)
        if len(column_to_loc) != num_columns:
            raise ValueError(
                f"Expected column names in '{cls.__name__}' to be unique"
            )

        out = Tensor._make_wrapper_subclass(
            cls,
            size=(*size, num_columns),
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
