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
    _data: torch.Tensor
    _names: tuple[str, ...]
    _stypes: tuple[Stype, ...]

    # Prevent auto-wrapping outputs back into the proper subclass type:
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore

    def __init__(
        self,
        data: Tensor,
        names: Sequence[str],
        stypes: Sequence[StypeLike],
    ) -> None:
        pass

    @staticmethod
    def __new__(
        cls,
        data: Tensor,
        names: Sequence[str],
        stypes: Sequence[StypeLike],
    ) -> "TableTensor":
        if isinstance(data, cls):  # If passed `TableTensor`, inherit metadata:
            data = data._data

        names = tuple(names)
        stypes = tuple(Stype(stype) for stype in stypes)

        if data.dim() != 2:
            raise ValueError(
                f"'{cls.__name__}' must be two-dimensional "
                f"(got {{data.dim()}})"
            )

        if len(names) != len(stypes):
            raise ValueError(
                f"The number of column names (got {len(names)}) must match "
                f"the number of semantic types (got {len(stypes)})"
            )

        out = torch.Tensor._make_wrapper_subclass(
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

        return out

    def as_tensor(self) -> torch.Tensor:
        return self._data

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def stypes(self) -> tuple[Stype, ...]:
        return self._stypes

    # PyTorch/Python builtins #################################################

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ["_data"]
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
        kwargs = pytree.tree_map_only(TableTensor, lambda x: x._data, kwargs)
        return func(*args, **(kwargs or {}))
