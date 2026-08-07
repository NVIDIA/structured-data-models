from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch

from sdm import RelatedTables, TableTensor
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    Recipe,
)
from sdm.tensor import EnsembleTable


@dataclass(frozen=True)
class _MemberContext:
    """Transformed context tables for one ensemble member."""

    x: TableTensor
    y: TableTensor
    related_tables: RelatedTables | None


@dataclass(frozen=True)
class _MemberQuery:
    """Transformed query tables for one ensemble member."""

    x: TableTensor
    related_tables: RelatedTables | None


class _RecipeExecution:
    """Bound recipe state for one ensemble preprocessing pass.

    Internal helper for :class:`~sdm.models.base.ICLModel`. Construct via
    :meth:`~sdm.processing.recipe.Recipe.bind`, then :meth:`transform`,
    :meth:`inverse_transform_target`, and :meth:`transform_output`.

    Attributes:
        recipe: Recipe with fitted ``features`` and ``target``.
        contexts: Transformed context tables, one per member.
    """

    def __init__(
        self,
        *,
        recipe: Recipe,
        contexts: Sequence[_MemberContext],
        related_processors: Mapping[str, EnsembleProcessor] | None,
    ) -> None:
        self.recipe = recipe
        self.contexts = tuple(contexts)  # TODO Do not cache!
        self._related_processors = related_processors

    @classmethod
    def _bind(
        cls,
        recipe: Recipe,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables | None,
        num_members: int,
        generator: torch.Generator | None,
    ) -> _RecipeExecution:
        """Bind a recipe to context data and return the execution state."""
        # Resolve task type first:
        y_ensemble = recipe.target.fit_transform_ensemble(
            EnsembleTable(y, num_members=num_members),
            generator=generator,
        )

        related_processors: Mapping[str, EnsembleProcessor] = {}
        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            related_processors = {}
            for name, table in related_tables.tables.items():
                processor = copy.deepcopy(recipe.features)
                related_processors[name] = processor
                related_ensembles[name] = processor.fit_transform_ensemble(
                    EnsembleTable(table, num_members=num_members),
                    generator=generator,
                )

        x_ensemble = recipe.features.fit_transform_ensemble(
            EnsembleTable(x, num_members=num_members),
            generator=generator,
        )

        contexts: Sequence[_MemberContext] = []
        for member_id in range(num_members):
            related_tables_i: RelatedTables | None = None
            if related_tables is not None:
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: table.table(member_id)
                        for name, table in related_ensembles.items()
                    },
                )
            contexts.append(
                _MemberContext(
                    x=x_ensemble.table(member_id),
                    y=y_ensemble.table(member_id),
                    related_tables=related_tables_i,
                )
            )
        return cls(
            recipe=recipe,
            contexts=tuple(contexts),
            related_processors=related_processors or None,
        )

    def transform(
        self,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[_MemberQuery, ...]:
        """Transform query features with the fitted processors.

        Args:
            x: Feature table for query examples.
            related_tables: Related query tables, or ``None``.
        """
        num_members = len(self.contexts)
        x_ensemble = self.recipe.features.transform_ensemble(
            EnsembleTable(x, num_members=num_members)
        )

        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            assert self._related_processors is not None
            for name, table in related_tables.tables.items():
                processor = self._related_processors[name]
                related_ensembles[name] = processor.transform_ensemble(
                    EnsembleTable(table, num_members=num_members)
                )

        queries: Sequence[_MemberQuery] = []
        for member_id in range(num_members):
            related_tables_i: RelatedTables | None = None
            if related_tables is not None:
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: table.table(member_id)
                        for name, table in related_ensembles.items()
                    },
                )
            queries.append(
                _MemberQuery(
                    x=x_ensemble.table(member_id),
                    related_tables=related_tables_i,
                )
            )
        return tuple(queries)

    def inverse_transform_target(
        self,
        outputs: Sequence[TableTensor],
    ) -> tuple[TableTensor, ...]:
        """Invert fitted target transforms on member outputs.

        Args:
            outputs: One model output per ensemble member.
        """
        num_members = len(self.contexts)
        if len(outputs) != num_members:
            raise ValueError(
                f"Expected {num_members} member outputs (got {len(outputs)})"
            )
        assert isinstance(self.recipe.target, EnsembleInvertibleMixin)
        table = self.recipe.target.inverse_transform_ensemble(
            EnsembleTable.from_tables(
                tables=outputs,
                member_table_ids=tuple(range(num_members)),
            )
        )
        return tuple(table.table(i) for i in range(table.num_members))

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        """Apply ``recipe.output`` to member outputs.

        Args:
            outputs: One model output per ensemble member.
        """
        if len(outputs) == 1:
            out = outputs[0].unsqueeze(0)
        else:
            out = torch.stack(list(outputs), dim=0)

        return self.recipe.output.transform(cast(TableTensor, out))
