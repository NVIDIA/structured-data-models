from __future__ import annotations

from typing import Any, Callable
from typing import Iterable, Sequence

import torch
import torch.utils._pytree as pytree

from schemafm.stype import Stype

aten = torch.ops.aten

HANDLED_FUNCTIONS: dict[Callable[..., Any], Callable[..., Any]] = {}

ColumnSelector = str | int | slice | Sequence[str] | Sequence[int]


def implements(torch_function: Callable[..., Any]) -> Callable[..., Any]:
    def decorator(my_function: Callable[..., Any]) -> Callable[..., Any]:
        HANDLED_FUNCTIONS[torch_function] = my_function
        return my_function

    return decorator


class TableTensor(torch.Tensor):
    _data: torch.Tensor
    _names: tuple[str, ...]
    _stypes: tuple[Stype, ...]
    _mask: torch.Tensor | None

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        names: Iterable[str] | None = None,
        stypes: Iterable[Stype | str] | None = None,
        mask: torch.Tensor | None = None,
    ) -> 'TableTensor':
        if isinstance(data, TableTensor):
            mask = data._mask if mask is None else mask
            names = data._names if names is None else names
            stypes = data._stypes if stypes is None else stypes
            data = data._data

        if names is None or stypes is None:
            raise TypeError("'names' and 'stypes' must be specified")

        names = tuple(names)
        stypes = tuple(Stype(stype) for stype in stypes)

        if not isinstance(data, torch.Tensor):
            raise TypeError("'data' must be a torch.Tensor")

        if data.layout != torch.strided:
            raise ValueError("'TableTensor' only supports strided tensors")

        if data.dim() not in (1, 2):
            raise ValueError(
                "'TableTensor' must be one- or two-dimensional",
            )

        if len(names) != len(stypes):
            raise ValueError(
                "The number of column names must match the number of "
                "semantic types",
            )

        if len(set(names)) != len(names):
            raise ValueError("Column names must be unique")

        if data.dim() == 1 and len(names) != 1:
            raise ValueError(
                "A one-dimensional 'TableTensor' must have exactly one "
                "column name and semantic type",
            )

        if data.dim() == 2 and data.size(1) != len(names):
            raise ValueError(
                "The last tensor dimension must match the number of named "
                "columns",
            )

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                raise TypeError("'mask' must be a torch.Tensor")
            if mask.dtype != torch.bool:
                raise ValueError("'mask' must have dtype torch.bool")
            if mask.shape != data.shape:
                raise ValueError("'mask' must have the same shape as 'data'")

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
        out._mask = mask
        return out

    @property
    def values(self) -> torch.Tensor:
        return self._data

    def as_tensor(self) -> torch.Tensor:
        return self._data

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def stypes(self) -> tuple[Stype, ...]:
        return self._stypes

    @property
    def mask(self) -> torch.Tensor | None:
        return self._mask

    def column_index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def indices_for(self, *stypes: Stype | str) -> tuple[int, ...]:
        stypes = tuple(Stype(stype) for stype in stypes)
        return tuple(
            index
            for index, stype in enumerate(self._stypes)
            if stype in stypes
        )

    def select(self, columns: ColumnSelector | torch.Tensor) -> 'TableTensor':
        if self.dim() != 2:
            raise ValueError(
                "Column selection requires a two-dimensional table",
            )

        indices = self._column_indices(columns)
        data = self._data[:, indices]
        mask = None if self._mask is None else self._mask[:, indices]
        names, stypes = self._select_metadata(indices)
        return TableTensor(data, names, stypes, mask)

    def drop(self, columns: ColumnSelector | torch.Tensor) -> 'TableTensor':
        if self.dim() != 2:
            raise ValueError("Column dropping requires a two-dimensional table")

        drop_indices = set(self._column_indices(columns))
        indices = tuple(
            index
            for index in range(len(self._names))
            if index not in drop_indices
        )
        data = self._data[:, indices]
        mask = None if self._mask is None else self._mask[:, indices]
        names, stypes = self._select_metadata(indices)
        return TableTensor(data, names, stypes, mask)

    def __getitem__(self, index: Any) -> torch.Tensor | 'TableTensor':
        if isinstance(index, str):
            return self[:, index]

        if isinstance(index, tuple):
            return self._getitem_tuple(index)

        data = self._data[index]
        mask = None if self._mask is None else self._mask[index]

        if data.dim() == self.dim():
            return TableTensor(data, self._names, self._stypes, mask)

        return data

    def _getitem_tuple(
        self,
        index: tuple[Any, ...],
    ) -> torch.Tensor | 'TableTensor':
        if len(index) == 2 and index[0] is Ellipsis:
            index = (slice(None), index[1])

        if len(index) != 2:
            data = self._data[index]
            return data

        row_index, column_index = index
        if self.dim() != 2:
            data = self._data[index]
            return data

        has_column_selector = _is_column_selector(column_index)
        if not has_column_selector:
            data = self._data[index]
            mask = None if self._mask is None else self._mask[index]
            if data.dim() == self.dim():
                return TableTensor(data, self._names, self._stypes, mask)
            return data

        data_column_index = self._data_column_index(column_index)
        data = self._data[row_index, data_column_index]
        mask = (
            None
            if self._mask is None
            else self._mask[row_index, data_column_index]
        )

        if _is_row_scalar(row_index):
            return data

        indices = self._column_indices(column_index)
        names, stypes = self._select_metadata(indices)
        return TableTensor(data, names, stypes, mask)

    def _column_indices(
        self,
        columns: ColumnSelector | torch.Tensor,
    ) -> tuple[int, ...]:
        if isinstance(columns, str):
            return (self.column_index(columns),)

        if isinstance(columns, int):
            return (self._normalize_column_index(columns),)

        if isinstance(columns, slice):
            return tuple(range(len(self._names))[columns])

        if isinstance(columns, torch.Tensor):
            if columns.dtype == torch.bool:
                if columns.numel() != len(self._names):
                    raise IndexError("Boolean column mask has invalid length")
                return tuple(
                    index
                    for index, keep in enumerate(columns.tolist())
                    if keep
                )

            return tuple(
                self._normalize_column_index(int(index))
                for index in columns.tolist()
            )

        values = tuple(columns)
        if len(values) == 0:
            return ()

        if all(isinstance(value, str) for value in values):
            return tuple(self.column_index(value) for value in values)

        if all(isinstance(value, bool) for value in values):
            if len(values) != len(self._names):
                raise IndexError("Boolean column mask has invalid length")
            return tuple(
                index
                for index, keep in enumerate(values)
                if keep
            )

        if all(isinstance(value, int) for value in values):
            return tuple(
                self._normalize_column_index(value)  # type: ignore[arg-type]
                for value in values
            )

        raise TypeError(f"Unsupported column selector: {columns!r}")

    def _normalize_column_index(self, index: int) -> int:
        if index < 0:
            index += len(self._names)

        if index < 0 or index >= len(self._names):
            raise IndexError("Column index out of range")

        return index

    def _select_metadata(
        self,
        indices: Sequence[int],
    ) -> tuple[tuple[str, ...], tuple[Stype, ...]]:
        return (
            tuple(self._names[index] for index in indices),
            tuple(self._stypes[index] for index in indices),
        )

    def _data_column_index(self, index: Any) -> Any:
        if isinstance(index, str):
            return self.column_index(index)

        if isinstance(index, (list, tuple)) and all(
            isinstance(value, str)
            for value in index
        ):
            return list(self._column_indices(index))

        return index

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        attrs = ['_data']
        if self._mask is not None:
            attrs.append('_mask')
        return attrs, (self._names, self._stypes)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        ctx: tuple[Any, ...],
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> 'TableTensor':
        del outer_size, outer_stride
        return TableTensor(
            inner_tensors['_data'],
            ctx[0],
            ctx[1],
            inner_tensors.get('_mask', None),
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable[..., Any],
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **(kwargs or {}))

        args = pytree.tree_map_only(TableTensor, lambda x: x._data, args)
        kwargs = pytree.tree_map_only(
            TableTensor,
            lambda x: x._data,
            kwargs,
        )
        return func(*args, **(kwargs or {}))

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'{self._data!r}, names={self._names!r}, '
            f'stypes={self._stypes!r})'
        )


def _is_column_selector(index: Any) -> bool:
    if isinstance(index, (str, int, slice, torch.Tensor)):
        return True

    if isinstance(index, (list, tuple)):
        return all(
            isinstance(value, (str, int, bool))
            for value in index
        )

    return False


def _is_row_scalar(index: Any) -> bool:
    if isinstance(index, int):
        return True

    if isinstance(index, torch.Tensor):
        return index.dim() == 0

    return False


def _wrap_like(
    input: TableTensor,
    data: torch.Tensor,
    mask: torch.Tensor | None,
    names: tuple[str, ...] | None = None,
    stypes: tuple[Stype, ...] | None = None,
) -> TableTensor | torch.Tensor:
    names = input.column_names if names is None else names
    stypes = input.stypes if stypes is None else stypes

    if data.dim() == 1 and len(names) == 1:
        return TableTensor(data, names, stypes, mask)

    if data.dim() == 2 and data.size(1) == len(names):
        return TableTensor(data, names, stypes, mask)

    return data


def _map_mask(
    mask: torch.Tensor | None,
    fn: Callable[..., torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor | None:
    if mask is None:
        return None
    return fn(mask, *args, **kwargs)


@implements(aten.clone.default)
def _clone(
    input: TableTensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> TableTensor:
    return TableTensor(
        input._data.clone(memory_format=memory_format),
        input.column_names,
        input.stypes,
        _map_mask(input._mask, torch.Tensor.clone),
    )


@implements(aten.detach.default)
def _detach(input: TableTensor) -> TableTensor:
    return TableTensor(
        input._data.detach(),
        input.column_names,
        input.stypes,
        _map_mask(input._mask, torch.Tensor.detach),
    )


@implements(aten.alias.default)
def _alias(input: TableTensor) -> TableTensor:
    return TableTensor(
        input._data,
        input.column_names,
        input.stypes,
        input._mask,
    )


@implements(aten._to_copy.default)
def _to_copy(
    input: TableTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> TableTensor:
    data = aten._to_copy.default(
        input._data,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )

    mask = input._mask
    if mask is not None:
        mask = mask.to(
            device=device,
            non_blocking=non_blocking,
        )

    return TableTensor(data, input.column_names, input.stypes, mask)


@implements(aten.cat.default)
def _cat(
    tensors: list[torch.Tensor | TableTensor],
    dim: int = 0,
) -> torch.Tensor | TableTensor:
    data_list = pytree.tree_map_only(
        TableTensor,
        lambda tensor: tensor._data,
        tensors,
    )
    data = aten.cat.default(data_list, dim=dim)

    if dim != 0:
        return data

    if not all(isinstance(tensor, TableTensor) for tensor in tensors):
        return data

    table_tensors = [
        tensor for tensor in tensors if isinstance(tensor, TableTensor)
    ]
    names = table_tensors[0].column_names
    stypes = table_tensors[0].stypes
    if any(tensor.column_names != names for tensor in table_tensors):
        return data
    if any(tensor.stypes != stypes for tensor in table_tensors):
        return data

    if any(tensor.mask is None for tensor in table_tensors):
        mask = None
    else:
        masks = [tensor.mask for tensor in table_tensors]
        assert all(mask is not None for mask in masks)
        mask = aten.cat.default(masks, dim=dim)

    return _wrap_like(table_tensors[0], data, mask, names, stypes)
