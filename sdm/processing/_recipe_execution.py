from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.processing.ensemble import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.processing.output.reduce import ReduceEstimators
from sdm.processing.recipe import Recipe
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
        self.contexts = tuple(contexts)
        self._related_processors = related_processors

    @classmethod
    def _bind(
        cls,
        *,
        recipe: Recipe,
        x_context: TableTensor,
        y_context: TableTensor,
        related_context_tables: RelatedTables | None,
        num_members: int,
        generator: torch.Generator | None,
    ) -> _RecipeExecution:
        """Bind a recipe to context data and return the execution state."""
        recipe = copy.deepcopy(recipe)

        related_context_out: dict[str, EnsembleTable] = {}
        related_processors: dict[str, EnsembleProcessor] = {}
        if related_context_tables is not None:
            for table_name, table in related_context_tables.tables.items():
                processor = cast(
                    EnsembleProcessor,
                    copy.deepcopy(recipe.features),
                )
                related_processors[table_name] = processor
                related_context_out[table_name] = (
                    processor.fit_transform_ensemble(
                        EnsembleTable(table, num_members=num_members),
                        generator=generator,
                    )
                )

        x_ensemble = EnsembleTable(x_context, num_members=num_members)
        y_ensemble = EnsembleTable(y_context, num_members=num_members)
        features = cast(EnsembleProcessor, recipe.features)
        target = cast(EnsembleProcessor, recipe.target)
        x_context_out = features.fit_transform_ensemble(
            x_ensemble,
            generator=generator,
        )
        y_context_out = target.fit_transform_ensemble(
            y_ensemble,
            generator=generator,
        )

        contexts = []
        for member_id in range(num_members):
            related_tables = None
            if related_context_tables is not None:
                related_tables = replace(
                    related_context_tables,
                    tables={
                        table_name: table.table(member_id)
                        for table_name, table in related_context_out.items()
                    },
                )
            contexts.append(
                _MemberContext(
                    x=x_context_out.table(member_id),
                    y=y_context_out.table(member_id),
                    related_tables=related_tables,
                )
            )
        return cls(
            recipe=recipe,
            contexts=tuple(contexts),
            related_processors=related_processors or None,
        )

    # test comment
    def transform(
        self,
        *,
        x_query: TableTensor,
        related_query_tables: RelatedTables | None,
    ) -> tuple[_MemberQuery, ...]:
        """Transform query features with the fitted processors.

        Args:
            x_query: Feature table for query examples.
            related_query_tables: Related query tables, or ``None``.
        """
        num_members = len(self.contexts)
        x_query_out = cast(
            EnsembleProcessor,
            self.recipe.features,
        ).transform_ensemble(EnsembleTable(x_query, num_members=num_members))

        related_query_out: dict[str, EnsembleTable] | None = None
        if related_query_tables is not None:
            related_query_out = {}
            related_processors = cast(
                Mapping[str, EnsembleProcessor],
                self._related_processors,
            )
            for table_name, table in related_query_tables.tables.items():
                related_query_out[table_name] = related_processors[
                    table_name
                ].transform_ensemble(
                    EnsembleTable(table, num_members=num_members)
                )

        queries = []
        for member_id in range(num_members):
            related_tables = None
            if related_query_tables is not None:
                assert related_query_out is not None
                related_tables = replace(
                    related_query_tables,
                    tables={
                        table_name: table.table(member_id)
                        for table_name, table in related_query_out.items()
                    },
                )
            queries.append(
                _MemberQuery(
                    x=x_query_out.table(member_id),
                    related_tables=related_tables,
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
        table = cast(
            EnsembleInvertibleMixin,
            self.recipe.target,
        ).inverse_transform_ensemble(
            EnsembleTable.from_tables(
                tables=outputs,
                member_table_ids=tuple(range(num_members)),
            )
        )
        return tuple(
            table.table(member_id) for member_id in range(table.num_members)
        )

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        """Apply ``recipe.output`` to member outputs.

        Stacks members on dim 0 unless the output pipeline reduces the
        ensemble dimension (e.g. :class:`~sdm.processing.ReduceEstimators`).

        Args:
            outputs: One model output per ensemble member.
        """
        num_members = len(outputs)
        table = EnsembleTable.from_tables(
            tables=outputs,
            member_table_ids=tuple(range(num_members)),
        )
        input_members = table.num_members
        table = cast(
            EnsembleProcessor,
            self.recipe.output,
        ).transform_ensemble(table)

        reduced = table.num_members < input_members or (
            table.num_members == 1
            and any(
                isinstance(module, ReduceEstimators)
                for module in self.recipe.output.modules()
            )
        )
        if reduced:
            return table.table(0)

        members = [
            table.table(member_id) for member_id in range(table.num_members)
        ]
        return cast(
            TableTensor,
            torch.stack(cast(list[Tensor], members), dim=0),
        )
