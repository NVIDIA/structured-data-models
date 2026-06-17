from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch import Tensor

from schemafm.column import Stype, StypeLike

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}


def implements(torch_function: Callable[..., Any]) -> Callable[..., Any]:

    def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


class TableTensor(Tensor):
    _data: Tensor
    _names: tuple[str, ...]
    _stypes: tuple[Stype, ...]
    _colptr: Tensor

    # Prevent auto-wrapping outputs back into the proper subclass type:
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    def __init__(
        self,
        data: Tensor,
        names: Sequence[str],
        stypes: Sequence[StypeLike],
        colptr: Tensor | Sequence[int] | None = None,
    ) -> None:
        pass

    @staticmethod
    def __new__(
        cls,
        data: Tensor,
        names: Sequence[str],
        stypes: Sequence[StypeLike],
        colptr: Tensor | Sequence[int] | None = None,
    ) -> "TableTensor":
        if isinstance(data, cls):
            data = data._data

        names, stypes = _normalize_metadata(names=names, stypes=stypes)
        colptr = _normalize_colptr(
            colptr=colptr,
            num_logical_columns=len(names),
        )

        if data.dim() != 2:
            raise ValueError(
                f"'{cls.__name__}' must be two-dimensional (got {data.dim()})"
            )

        if data.size(1) != int(colptr[-1]):
            raise ValueError(
                f"The number of physical columns (got {data.size(1)}) must "
                f"match 'colptr[-1]' (got {int(colptr[-1])})"
            )

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
        out._names = names
        out._stypes = stypes
        out._colptr = colptr

        return out

    def as_tensor(self) -> Tensor:
        return self._data

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def stypes(self) -> tuple[Stype, ...]:
        return self._stypes

    @property
    def colptr(self) -> Tensor:
        return self._colptr

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data", "_colptr"]
        ctx = (self._names, self._stypes)
        return attrs, ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> "TableTensor":
        names, stypes = ctx
        return TableTensor(
            data=inner_tensors["_data"],
            names=names,
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
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        args = pytree.tree_map_only(TableTensor, lambda x: x._data, args)
        if kwargs is not None:
            kwargs = pytree.tree_map_only(
                TableTensor,
                lambda x: x._data,
                kwargs,
            )
        return func(*args, **(kwargs or {}))


def _normalize_metadata(
    *,
    names: Sequence[str],
    stypes: Sequence[StypeLike],
) -> tuple[tuple[str, ...], tuple[Stype, ...]]:
    if len(names) != len(stypes):
        raise ValueError(
            f"The number of column names (got {len(names)}) must match "
            f"the number of semantic types (got {len(stypes)})"
        )

    names = tuple(names)
    stypes = tuple(Stype(stype) for stype in stypes)

    if len(set(names)) != len(names):
        raise ValueError("Column names must be unique")

    return names, stypes


def _normalize_colptr(
    *,
    colptr: Tensor | Sequence[int] | None,
    num_logical_columns: int,
) -> Tensor:
    if colptr is None:
        return torch.arange(num_logical_columns + 1, dtype=torch.long)

    if isinstance(colptr, Tensor):
        colptr = colptr.detach().to(dtype=torch.long, device="cpu")
    else:
        colptr = torch.tensor(colptr, dtype=torch.long)

    if colptr.dim() != 1:
        raise ValueError("'colptr' must be one-dimensional")

    if colptr.numel() != num_logical_columns + 1:
        raise ValueError(
            f"'colptr' must have length {num_logical_columns + 1} "
            f"(got {colptr.numel()})"
        )

    if colptr.numel() == 0 or int(colptr[0]) != 0:
        raise ValueError("'colptr' must start with 0")

    if (colptr[:-1] >= colptr[1:]).any():
        raise ValueError("'colptr' must be strictly increasing")

    return colptr
