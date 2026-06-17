from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import Tensor

from schemafm import Stype, StypeLike

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(torch_function: Callable[..., Any]) -> Callable[..., Any]:

    def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


class TableTensor(Tensor):
    _data: Tensor
    _col_names: tuple[str, ...]
    _stypes: tuple[Stype, ...]
    _colptr: Tensor

    # Prevent auto-wrapping outputs back into the proper subclass type:
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    def __init__(
        self,
        data: Tensor,
        col_names: Sequence[str],
        stypes: Sequence[StypeLike],
        colptr: Tensor | Sequence[int],
    ) -> None:
        pass

    @staticmethod
    def __new__(
        cls,
        data: Tensor,
        col_names: Sequence[str],
        stypes: Sequence[StypeLike],
        colptr: Tensor | Sequence[int],
    ) -> "TableTensor":

        col_names = tuple(col_names)
        stypes = tuple(Stype(stype) for stype in stypes)
        colptr = torch.as_tensor(colptr, dtype=torch.long, device=data.device)

        if data.dim() != 2:
            raise ValueError(
                f"Expected 'data' in '{cls.__name__}' to be two-dimensional "
                f"(got {data.dim()}D tensor)"
            )

        if colptr.dim() != 1:
            raise ValueError(
                f"Expected 'colptr' in '{cls.__name__}' to be one-dimensional "
                f"(got {colptr.dim()}D tensor)"
            )

        if len(col_names) != colptr.numel() - 1:
            raise ValueError(
                f"The number of column names must match the number of "
                f"logical columns (got {len(col_names)}, but expected "
                f"{colptr.numel() - 1})"
            )

        if len(stypes) != colptr.numel() - 1:
            raise ValueError(
                f"The number of semantic types must match the number of "
                f"logical columns (got {len(stypes)}, but expected "
                f"{colptr.numel() - 1})"
            )

        if len(set(col_names)) != len(col_names):
            raise ValueError("Column names must be unique")

        out = Tensor._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=data.requires_grad,
        )

        out._data = data
        out._col_names = col_names
        out._stypes = stypes
        out._colptr = colptr

        return out

    def as_tensor(self) -> Tensor:
        return self._data

    @property
    def col_names(self) -> tuple[str, ...]:
        return self._col_names

    @property
    def stypes(self) -> tuple[Stype, ...]:
        return self._stypes

    @property
    def colptr(self) -> Tensor:
        return self._colptr

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data", "_colptr"]
        ctx = (self._col_names, self._stypes)
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> "TableTensor":
        col_names, stypes = ctx
        return TableTensor(
            data=inner_tensors["_data"],
            col_names=col_names,
            stypes=stypes,
            colptr=inner_tensors["_colptr"],
        )

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        print(func.__name__)
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        args = pytree.tree_map_only(TableTensor, lambda x: x._data, args)
        kwargs = pytree.tree_map_only(TableTensor, lambda x: x._data, kwargs)
        return func(*args, **(kwargs or {}))
