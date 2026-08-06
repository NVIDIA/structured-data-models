from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch
from torch import Tensor

from sdm import RelatedTables, TableTensor
from sdm.processing.ensemble import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.processing.recipe import Recipe
from sdm.tensor import EnsembleTable


@dataclass(frozen=True)
class _ContextInput:
    x: TableTensor
    y: TableTensor
    related_tables: RelatedTables | None


@dataclass(frozen=True)
class _QueryInput:
    x: TableTensor
    related_tables: RelatedTables | None


class _RecipeExecution:
    """Batched recipe preprocessing for ensemble models.

    Fits and applies :class:`~sdm.processing.recipe.Recipe` steps (features,
    target, output) across all ensemble members in a single pass using
    :class:`~sdm.tensor.EnsembleTable`, rather than looping over members
    sequentially.

    Use :meth:`fit_context` to create an instance from raw context data,
    :meth:`transform_query` to preprocess unseen query data with the
    fitted state, and :meth:`transform_output` to apply inverse-target
    and output postprocessing to the stacked member predictions.

    Args:
        recipe: Deepcopied recipe whose processors hold fitted state.
        context_inputs: Per-member preprocessed context tables.
        classes: Per-member class labels, or ``None`` for regression members.
        is_regression: Whether the target is regression (no categorical
            columns).
        related_processors: Per-related-table fitted feature processors,
            or ``None`` when no related tables are used.
    """

    def __init__(
        self,
        *,
        recipe: Recipe,
        context_inputs: Sequence[_ContextInput],
        classes: Sequence[Tensor | None],
        is_regression: bool,
        related_processors: Mapping[str, EnsembleProcessor] | None,
    ) -> None:
        self.recipe = recipe
        self.context_inputs = tuple(context_inputs)
        self.classes = tuple(classes)
        self.is_regression = is_regression
        self._related_processors = related_processors

    @classmethod
    def fit_context(
        cls,
        *,
        recipe: Recipe,
        x_context: TableTensor,
        y_context: TableTensor,
        related_context_tables: RelatedTables | None,
        num_members: int,
        generator: torch.Generator | None,
    ) -> _RecipeExecution:
        """Fit recipe processors on context data and return execution state.

        Deepcopies the recipe, fits feature and target processors on
        ensemble-wrapped context tables, and extracts per-member
        preprocessed inputs. Related tables are fitted with independent
        copies of the feature processor taken before the main fit.

        Args:
            recipe: Recipe to deepcopy and fit.
            x_context: Feature table for in-context examples.
            y_context: Target table for in-context examples.
            related_context_tables: Related context tables, or ``None``.
            num_members: Number of ensemble members.
            generator: Pseudorandom number generator for sampling.
        """
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

        is_regression = y_context.categorical.size(-1) == 0

        context_inputs = []
        classes = []
        for member_id in range(num_members):
            y_i = y_context_out.table(member_id)
            context_inputs.append(
                _ContextInput(
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
                None if is_regression else y_i.categorical.categories[0]
            )

        return cls(
            recipe=recipe,
            context_inputs=tuple(context_inputs),
            classes=tuple(classes),
            is_regression=is_regression,
            related_processors=related_processors or None,
        )

    def transform_query(
        self,
        *,
        x_query: TableTensor,
        related_query_tables: RelatedTables | None,
    ) -> tuple[_QueryInput, ...]:
        """Transform query features using the fitted recipe processors.

        Args:
            x_query: Feature table for query examples.
            related_query_tables: Related query tables, or ``None``.
        """
        num_members = len(self.context_inputs)
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
            _QueryInput(
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
        """Apply inverse-target and output postprocessing to member outputs.

        For regression tasks, applies the fitted target's inverse
        transform before the output processor. Returns a single table
        when ``num_members`` is one, otherwise stacks members along a
        leading dimension.

        Args:
            outputs: One raw model output per ensemble member.
        """
        num_members = len(self.context_inputs)
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
        table = cast(
            EnsembleProcessor,
            recipe.output,
        ).transform_ensemble(table)
        if table.num_members == 1:
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
