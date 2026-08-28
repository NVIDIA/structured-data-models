from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

import torch

import sdm.processing as sp
from sdm import Recipe, RelatedTables, TableTensor
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable


@dataclass(frozen=True)
class MemberContext:
    """Transformed context tables for one ensemble member."""

    x: TableTensor
    y: TableTensor
    related_tables: RelatedTables[TableTensor] | None


@dataclass(frozen=True)
class MemberQuery:
    """Transformed query tables for one ensemble member."""

    x: TableTensor
    related_tables: RelatedTables[TableTensor] | None


class RecipeExecution:
    """Recipe execution manager during model processing."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

        self._related_processors: Mapping[str, EnsembleProcessor] | None = None
        self._target_locations: tuple[tuple[int, int], ...] | None = None

    def fit_transform(
        self,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
        *,
        num_members: int = 1,
        generator: torch.Generator | None = None,
    ) -> tuple[MemberContext, ...]:
        """Fit and transform context data."""
        # Inverse target transform requires distinct member assignment:
        y_ensemble = EnsembleTable.__new__(EnsembleTable)
        y_ensemble._groups = (
            cast(TableTensor, y.unsqueeze(0).expand(num_members, *y.size())),
        )
        y_ensemble._locations = tuple((0, i) for i in range(num_members))

        # Transform target first to be able to resolve task type:
        y_ensemble = self.recipe.target.fit_transform_ensemble(
            y_ensemble,
            generator=generator,
        )
        self._target_locations = y_ensemble._locations

        task_dispatchers = tuple(
            module
            for processor in (self.recipe.features, self.recipe.output)
            for module in processor.modules()
            if isinstance(module, sp.TaskDispatch)
        )
        if len(task_dispatchers) > 0:
            if any(group.size(-1) != 1 for group in y_ensemble):
                raise ValueError(
                    "Expected the transformed target to contain exactly one "
                    "column"
                )

            if all(group.numerical.size(-1) == 1 for group in y_ensemble):
                task = "regression"
            elif all(group.categorical.size(-1) == 1 for group in y_ensemble):
                task = "classification"
            else:
                raise ValueError(
                    "'Recipe.target' must resolve to a single task type"
                )

            for task_dispatcher in task_dispatchers:
                task_dispatcher._task = task

        self._related_processors = None
        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            self._related_processors = {}
            for name, table in related_tables.tables.items():
                processor = copy.deepcopy(self.recipe.features)
                for module in processor.modules():
                    if isinstance(module, sp.TableDispatch):
                        module._route = "related"
                self._related_processors[name] = processor
                related_ensembles[name] = processor.fit_transform_ensemble(
                    EnsembleTable(table, num_members=num_members),
                    generator=generator,
                )

        for module in self.recipe.features.modules():
            if isinstance(module, sp.TableDispatch):
                module._route = "task"

        x_ensemble = self.recipe.features.fit_transform_ensemble(
            EnsembleTable(x, num_members=num_members),
            generator=generator,
        )

        members: list[MemberContext] = []
        for member_id in range(num_members):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: table.table(member_id)
                        for name, table in related_ensembles.items()
                    },
                )
            members.append(
                MemberContext(
                    x=x_ensemble.table(member_id),
                    y=y_ensemble.table(member_id),
                    related_tables=related_tables_i,
                )
            )

        return tuple(members)

    def transform(
        self,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[MemberQuery, ...]:
        """Transform query data."""
        assert self._target_locations is not None
        num_members = len(self._target_locations)
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

        members: list[MemberQuery] = []
        for member_id in range(num_members):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = replace(
                    related_tables,
                    tables={
                        name: table.table(member_id)
                        for name, table in related_ensembles.items()
                    },
                )
            members.append(
                MemberQuery(
                    x=x_ensemble.table(member_id),
                    related_tables=related_tables_i,
                )
            )

        return tuple(members)

    def inverse_transform_target(
        self,
        outputs: Sequence[TableTensor],
    ) -> tuple[TableTensor, ...]:
        """Invert fitted target transforms on member outputs."""
        # Reconstruct the group layout of the transformed target:
        assert self._target_locations is not None
        assert len(outputs) == len(self._target_locations)
        num_groups = max(group for group, _ in self._target_locations) + 1
        groups: list[list[TableTensor | None]] = [
            [] for _ in range(num_groups)
        ]
        for group_id, _ in self._target_locations:
            groups[group_id].append(None)
        for i, (group_id, position) in enumerate(self._target_locations):
            groups[group_id][position] = outputs[i]

        table = EnsembleTable.__new__(EnsembleTable)
        table._groups = tuple(
            cast(
                TableTensor,
                group[0].unsqueeze(0)  # type: ignore
                if len(group) == 1
                else torch.stack(group, dim=0),  # type: ignore
            )
            for group in groups
        )
        table._locations = self._target_locations

        if not isinstance(self.recipe.target, EnsembleInvertibleMixin):
            raise RuntimeError("Target recipe is not invertible")
        table = self.recipe.target.inverse_transform_ensemble(table)
        return tuple(table.table(i) for i in range(table.num_members))

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        """Apply ``recipe.output`` to member outputs."""
        if len(outputs) == 1:
            out = outputs[0].unsqueeze(0)
        else:
            out = torch.stack(list(outputs), dim=0)

        return self.recipe.output.transform(cast(TableTensor, out))
