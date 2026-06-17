from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import torch


class Stype(str, Enum):
    numerical = 'numerical'
    categorical = 'categorical'
    timestamp = 'timestamp'
    constant = 'constant'


ColumnSelector = str | int | slice | Sequence[str] | Sequence[int]


@dataclass(frozen=True)
class Schema:
    names: tuple[str, ...]
    stypes: tuple[Stype, ...]

    def __init__(
        self,
        names: Iterable[str],
        stypes: Iterable[Stype | str],
    ) -> None:
        names = tuple(names)
        stypes = tuple(Stype(stype) for stype in stypes)

        if len(names) != len(stypes):
            raise ValueError(
                "The number of column names must match the number of "
                "semantic types",
            )

        if len(set(names)) != len(names):
            raise ValueError("Column names must be unique")

        object.__setattr__(self, 'names', names)
        object.__setattr__(self, 'stypes', stypes)

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def indices_for(self, *stypes: Stype | str) -> tuple[int, ...]:
        stypes = tuple(Stype(stype) for stype in stypes)
        return tuple(
            index
            for index, stype in enumerate(self.stypes)
            if stype in stypes
        )

    def select(self, columns: ColumnSelector | torch.Tensor) -> 'Schema':
        indices = self._indices(columns)
        return Schema(
            names=tuple(self.names[index] for index in indices),
            stypes=tuple(self.stypes[index] for index in indices),
        )

    def drop(self, columns: ColumnSelector | torch.Tensor) -> 'Schema':
        drop_indices = set(self._indices(columns))
        keep_indices = tuple(
            index for index in range(len(self)) if index not in drop_indices
        )
        return self.select(keep_indices)

    def _indices(
        self,
        columns: ColumnSelector | torch.Tensor,
    ) -> tuple[int, ...]:
        if isinstance(columns, str):
            return (self.index(columns),)

        if isinstance(columns, int):
            return (self._normalize_index(columns),)

        if isinstance(columns, slice):
            return tuple(range(len(self))[columns])

        if isinstance(columns, torch.Tensor):
            if columns.dtype == torch.bool:
                if columns.numel() != len(self):
                    raise IndexError("Boolean column mask has invalid length")
                return tuple(
                    index
                    for index, keep in enumerate(columns.tolist())
                    if keep
                )

            return tuple(
                self._normalize_index(int(index))
                for index in columns.tolist()
            )

        values = tuple(columns)
        if len(values) == 0:
            return ()

        if all(isinstance(value, str) for value in values):
            return tuple(self.index(value) for value in values)

        if all(isinstance(value, bool) for value in values):
            if len(values) != len(self):
                raise IndexError("Boolean column mask has invalid length")
            return tuple(
                index
                for index, keep in enumerate(values)
                if keep
            )

        if all(isinstance(value, int) for value in values):
            return tuple(
                self._normalize_index(value)  # type: ignore[arg-type]
                for value in values
            )

        raise TypeError(f"Unsupported column selector: {columns!r}")

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError("Column index out of range")

        return index
