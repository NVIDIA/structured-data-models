from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from typing_extensions import Self

from sdm.stype import Stype
from sdm.tensor.table import TableTensor


@dataclass(frozen=True, init=False)
class EnsembleTable:
    r"""Store shared and distinct representations of one table.

    Args:
        table: The initial shared table with shape ``[..., R, C]``.
        num_estimators: Positive number of estimators.
    """

    _tables_storage: tuple[TableTensor, ...]
    _estimator_table_locations: dict[int, tuple[int, int]]

    def __init__(self, table: TableTensor, *, num_estimators: int) -> None:
        if num_estimators < 1:
            raise ValueError("'num_estimators' needs to be positive.")
        object.__setattr__(
            self,
            "_tables_storage",
            (cast(TableTensor, table.unsqueeze(0)),),
        )
        object.__setattr__(
            self,
            "_estimator_table_locations",
            dict.fromkeys(range(num_estimators), (0, 0)),
        )

    @classmethod
    def from_table_tensors(
        cls,
        estimator_to_table_map: Mapping[int, TableTensor],
    ) -> Self:
        r"""Build an ensemble table from estimator-specific tables.

        Tables with compatible metadata are stacked into shared storage, while
        incompatible tables remain in separate storage entries.

        Args:
            estimator_to_table_map: TableTensor for each estimator id.
        """
        if len(estimator_to_table_map) == 0:
            raise ValueError("Expected at least one estimator table.")

        # mapping between compatibility_key and tuple of
        # (group_index, tables_in_that_group)
        groups: dict[tuple[object, ...], tuple[int, list[TableTensor]]] = {}
        table_locations: dict[int, tuple[int, int]] = {}

        for estimator_id, table in estimator_to_table_map.items():
            # compatibility_key decides whether tables can be stacked together
            compatibility_key = (
                tuple(table.size()),
                tuple(
                    (stype, table.columns[stype], table.blocks[stype].dtype)
                    for stype in Stype
                ),
                tuple(
                    id(category) for category in table.categorical.categories
                ),
            )
            if compatibility_key not in groups:
                groups[compatibility_key] = (len(groups), [])
            group_index, group = groups[compatibility_key]
            table_locations[estimator_id] = (group_index, len(group))
            group.append(table)

        stacked_tables_storage = tuple(
            cast(TableTensor, group[0].unsqueeze(0))
            if len(group) == 1
            else cast(TableTensor, torch.stack(tuple(group), dim=0))
            for _, group in groups.values()
        )

        table = cls.__new__(cls)
        object.__setattr__(table, "_tables_storage", stacked_tables_storage)
        object.__setattr__(
            table,
            "_estimator_table_locations",
            table_locations,
        )
        return table

    def to_table_tensor(
        self,
        estimator_ids: Sequence[int] | None = None,
    ) -> TableTensor:
        r"""Materialize estimators with identical table metadata.

        Args:
            estimator_ids: Optional estimator ids. All estimators are used by
                default.
        """
        selected_estimator_ids = (
            self._estimator_table_locations.keys()
            if estimator_ids is None
            else estimator_ids
        )

        tables: list[TableTensor] = []
        reference_key: tuple[object, ...] | None = None
        already_materialized = len(self._tables_storage) == 1
        for index, estimator_id in enumerate(selected_estimator_ids):
            storage_index, table_index = self._estimator_table_locations[
                estimator_id
            ]
            if (storage_index, table_index) != (0, index):
                already_materialized = False
            table = self._tables_storage[storage_index][table_index]
            compatibility_key = (
                tuple(table.size()),
                tuple(
                    (stype, table.columns[stype], table.blocks[stype].dtype)
                    for stype in Stype
                ),
                tuple(
                    id(category) for category in table.categorical.categories
                ),
            )
            if reference_key is None:
                reference_key = compatibility_key
            elif compatibility_key != reference_key:
                raise ValueError(
                    "Cannot materialize estimators with different metadata."
                )
            tables.append(table)

        if already_materialized:
            return self._tables_storage[0]
        return cast(
            TableTensor,
            torch.stack(cast(list[torch.Tensor], tables), dim=0),
        )
