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
    :meth:`~sdm.processing.recipe.Recipe.bind`, then :meth:`transform` and
    :meth:`transform_output`.

    Attributes:
        recipe: Recipe with fitted ``features`` and ``target``.
        contexts: Transformed context tables, one per member.
        classes: Per-member class labels, or ``None`` for regression.
        is_regression: ``True`` when every member has a regression target.
    """

    def __init__(
        self,
        *,
        recipe: Recipe,
        contexts: Sequence[_MemberContext],
        classes: Sequence[Tensor | None],
        is_regression: bool,
        related_processors: Mapping[str, EnsembleProcessor] | None,
    ) -> None:
        self.recipe = recipe
        self.contexts = tuple(contexts)
        self.classes = tuple(classes)
        self.is_regression = is_regression
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
        classes = []
        for member_id in range(num_members):
            y_i = y_context_out.table(member_id)
            contexts.append(
                _MemberContext(
                    x=x_context_out.table(member_id),
                    y=y_i,
                    related_tables=cls._replace_with_member_tables(
                        related_context_tables,
                        related_context_out,
                        member_id,
                    ),
                )
            )
            classes.append(
                y_i.categorical.categories[0]
                if y_i.categorical.size(-1) > 0
                else None
            )

        is_regression = all(c is None for c in classes)

        return cls(
            recipe=recipe,
            contexts=tuple(contexts),
            classes=tuple(classes),
            is_regression=is_regression,
            related_processors=related_processors or None,
        )

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

        return tuple(
            _MemberQuery(
                x=x_query_out.table(member_id),
                related_tables=self._replace_with_member_tables(
                    related_query_tables,
                    related_query_out,
                    member_id,
                ),
            )
            for member_id in range(num_members)
        )

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        """Postprocess member outputs via inverse target (if regression) and
        ``recipe.output``.

        Stacks members on dim 0 unless the output pipeline reduces the
        ensemble dimension (e.g. :class:`~sdm.processing.ReduceEstimators`).

        Args:
            outputs: One raw model output per ensemble member.
        """
        num_members = len(self.contexts)
        if len(outputs) != num_members:
            raise ValueError(
                f"Expected {num_members} member outputs (got {len(outputs)})"
            )

        recipe = self.recipe
        table = EnsembleTable.from_tables(
            tables=outputs,
            member_table_ids=tuple(range(num_members)),
        )
        if self.is_regression:
            target = cast(EnsembleInvertibleMixin, recipe.target)
            table = target.inverse_transform_ensemble(table)
        input_members = table.num_members
        table = cast(
            EnsembleProcessor,
            recipe.output,
        ).transform_ensemble(table)

        reduced = table.num_members < input_members or (
            table.num_members == 1
            and any(
                isinstance(module, ReduceEstimators)
                for module in recipe.output.modules()
            )
        )
        if reduced:
            return table.table(0)

        members = [
            table.table(member_id) for member_id in range(table.num_members)
        ]
        return cast(
            TableTensor,
            torch.stack(cast(list[torch.Tensor], members), dim=0),
        )

    @staticmethod
    def _replace_with_member_tables(
        template: RelatedTables | None,
        tables: Mapping[str, EnsembleTable] | None,
        member_id: int,
    ) -> RelatedTables | None:
        if template is None or tables is None:
            return None
        return replace(
            template,
            tables={
                table_name: table.table(member_id)
                for table_name, table in tables.items()
            },
        )
