from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, NamedTuple, TypeVar

import torch
from torch import Tensor

from schemafm.stype import Stype, StypeLike
from schemafm.tensor.categorical import CategoricalTensor

aten = torch.ops.aten

SelfTableTensor = TypeVar("SelfTableTensor", bound="TableTensor")


class _ColumnIndex(NamedTuple):
    stype: Stype
    index: int


class TableTensor(Tensor):
    HANDLED_FUNCTIONS: ClassVar[
        dict[Callable[..., Any], Callable[..., Any]]
    ] = {}

    _numerical: Tensor
    _categorical: CategoricalTensor
    _columns: dict[Stype, tuple[str, ...]]
    _column_to_loc: dict[str, _ColumnIndex]

    # Route tensor operations through `__torch_dispatch__` only.
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    # Constructors ############################################################

    def __init__(
        cls,
        columns: Mapping[StypeLike, Sequence[str]] | None,
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

        size = size(tuple) if size is not None else size

        for block in (numerical, categorical):
            size = tuple(block.size()[:-1]) if size is None else size
            device = block.device if device is None else device

            # TODO Check size
            # TODO Check device

        if size is None:
            raise ValueError(
                f"Expected 'size' in '{cls.__name__}' to be given when "
                f"all blocks are 'None'"
            )

        if numerical is None:
            numerical = torch.empty((*size, 0), device=device)
        if categorical is None:
            categorical = CategoricalTensor(
                data=torch.empty((*size, 0), dtype=torch.int32, device=device),
                categories=(),
            )

        # TODO Build columns
        # TODO Build column loc
        # TODO Check unique column names
        # TODO compute global number of columns

        out = Tensor._make_wrapper_subclass(
            cls,
            size=(*size, num_columns),
            strides=_contiguous_stride((*size, num_columns)),
            storage_offset=0,
            dtype=torch.uint8,
            device=numerical.device,
            layout=torch.strided,
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


def _contiguous_stride(size: Sequence[int]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim_size in reversed(size):
        strides.append(stride)
        stride *= dim_size
    return tuple(reversed(strides))
