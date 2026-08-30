from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import NamedTuple, cast

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import Recipe, RelatedTables, TableTensor
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable


class MemberContext(NamedTuple):
    """Transformed context tables for one ensemble member."""

    x: TableTensor
    y: TableTensor
    related_tables: RelatedTables[TableTensor] | None


class MemberQuery(NamedTuple):
    """Transformed query tables for one ensemble member."""

    x: TableTensor
    related_tables: RelatedTables[TableTensor] | None


class RecipeExecution:
    """Recipe execution manager during model processing."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

        self._related_processors: Mapping[str, EnsembleProcessor] | None = None
        self._num_members: int | None = None
        self._y_locations: tuple[tuple[int, int], ...] | None = None

    def fit_transform(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        y: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables[TableTensor] | RelatedTables[EnsembleTable] | None,
        *,
        num_members: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[MemberContext, ...]:
        """Fit and transform context data."""
        # Transform target first to be able to resolve task type. Inverse
        # target transforms require distinct member assignment:
        y = _to_ensemble_tensor(y, num_members, expand=True)
        y = self.recipe.target.fit_transform_ensemble(y, generator=generator)

        self._num_members = num_members
        self._y_locations = y._locations

        task_dispatchers = tuple(
            module
            for processor in (self.recipe.features, self.recipe.output)
            for module in processor.modules()
            if isinstance(module, sp.TaskDispatch)
        )
        if len(task_dispatchers) > 0:
            if any(group.size(-1) != 1 for group in y):
                raise ValueError(
                    "Expected the transformed target to contain exactly one "
                    "column"
                )

            if all(group.numerical.size(-1) == 1 for group in y):
                task = "regression"
            elif all(group.categorical.size(-1) == 1 for group in y):
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
                    _to_ensemble_tensor(table, num_members, expand=False),
                    generator=generator,
                )

        for module in self.recipe.features.modules():
            if isinstance(module, sp.TableDispatch):
                module._route = "task"

        x = _to_ensemble_tensor(x, num_members, expand=False)
        x = self.recipe.features.fit_transform_ensemble(x, generator=generator)

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
                    x=x.table(member_id),
                    y=y.table(member_id),
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
        assert self._y_locations is not None

        x = _to_ensemble_tensor(x, self._num_members, expand=False)
        x = self.recipe.features.transform_ensemble(x)

        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            assert self._related_processors is not None
            for name, table in related_tables.tables.items():
                processor = self._related_processors[name]
                related_ensembles[name] = processor.transform_ensemble(
                    _to_ensemble_tensor(table, self._num_members, expand=False)
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
                    x=x.table(member_id),
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
        assert self._y_locations is not None
        assert len(outputs) == len(self._y_locations)
        num_groups = max(group for group, _ in self._y_locations) + 1
        groups: list[list[TableTensor | None]] = [
            [] for _ in range(num_groups)
        ]
        for group_id, _ in self._y_locations:
            groups[group_id].append(None)
        for i, (group_id, position) in enumerate(self._y_locations):
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
        table._locations = self._y_locations

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


def _to_ensemble_tensor(
    x: Tensor | TableTensor | EnsembleTable,
    num_estimators: int | None,
    *,
    expand: bool = False,
) -> EnsembleTable:
    r"""Convert model input data into an ensemble-aware representation.

    Args:
        x: The input table.
        num_estimators: Number of estimators to represent. If omitted, a 2D
            input creates one estimator and higher-rank inputs infer the
            estimator count from their leading dimension.
        expand: Whether to expand shared inputs into distinct logical member
            positions.
    """
    if isinstance(x, EnsembleTable):
        if num_estimators is not None and num_estimators != x.num_members:
            raise ValueError(
                f"Expected {num_estimators} members in 'EnsembleTable' "
                f"(got {x.num_members})"
            )
        if expand:
            groups = [x.expanded_group(i) for i in range(x.num_groups)]
            locations = []
            next_pos = [0] * x.num_groups
            for i, _ in x._locations:
                locations.append((i, next_pos[i]))
                next_pos[i] += 1
            x = EnsembleTable._from_groups(groups, locations)
        return x

    if not isinstance(x, TableTensor):
        x = TableTensor.from_tensor(x)

    # Treat leading dimension as ensemble dimension:
    if x.dim() > 2 and num_estimators is None:
        locations = tuple((0, i) for i in range(x.size(0)))
        return EnsembleTable._from_groups((x,), locations)

    num_estimators = 1 if num_estimators is None else num_estimators
    if num_estimators < 1:
        raise ValueError("'num_estimators' needs to be positive")

    # Add a leading ensemble dimension:
    x = x.unsqueeze(0)
    if expand:
        x = x.expand(num_estimators, *x.size()[1:])

    if x.size(0) == 1:
        locations = ((0, 0),) * num_estimators
    else:
        locations = tuple((0, i) for i in range(num_estimators))

    return EnsembleTable._from_groups((cast(TableTensor, x),), locations)
