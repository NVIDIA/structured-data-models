from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import NamedTuple, cast

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import Recipe, RelatedTables, Stype, TableTensor
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
        self._num_estimators: int | None = None
        self._partition_rows = False
        self._y_locations: tuple[tuple[int, int], ...] | None = None

    @property
    def num_members(self) -> int:
        r"""The number of fitted members."""
        assert self._y_locations is not None
        return len(self._y_locations)

    def fit_transform(
        self,
        x: Tensor | TableTensor | EnsembleTable,
        y: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None,
        *,
        num_members: int | None = None,
        generator: torch.Generator | None = None,
        partition_rows: bool = False,
    ) -> tuple[MemberContext, ...]:
        """Fit and transform context data."""
        partition_rows = (
            partition_rows
            and num_members is not None
            and num_members > 1
            and not isinstance(x, EnsembleTable)
            and not isinstance(y, EnsembleTable)
            and x.dim() == 2
            and y.dim() == 2
            # Preserve full contexts rather than create empty members.
            and x.size(-2) >= num_members
        )

        x = _to_ensemble_table(x, num_members)
        y = _to_ensemble_table(y, num_members, expand=True)
        # Transform target first to be able to resolve task type. Inverse
        # target transforms require distinct member assignment:
        y = self.recipe.target.fit_transform_ensemble(y, generator=generator)

        self._num_estimators = num_members
        self._partition_rows = partition_rows
        self._y_locations = y._locations

        if partition_rows:
            x, y = _partition_rows(x, y, generator=generator)

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
                    _to_ensemble_table(table, num_members),
                    generator=generator,
                )
                if related_ensembles[name].num_members != self.num_members:
                    raise ValueError(
                        "Expected inputs to map to the same number of "
                        "ensemble members"
                    )

        for module in self.recipe.features.modules():
            if isinstance(module, sp.TableDispatch):
                module._route = "task"

        x = self.recipe.features.fit_transform_ensemble(x, generator=generator)
        if x.num_members != self.num_members:
            raise ValueError(
                "Expected inputs to map to the same number of ensemble members"
            )

        members: list[MemberContext] = []
        for member_id in range(self.num_members):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = RelatedTables(
                    tables={
                        name: table.table(member_id)
                        for name, table in related_ensembles.items()
                    },
                    relationships=related_tables.relationships,
                    task_links=related_tables.task_links,
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
        x: Tensor | TableTensor | EnsembleTable,
        related_tables: RelatedTables | None,
    ) -> tuple[MemberQuery, ...]:
        """Transform query data."""
        x = _to_ensemble_table(
            x,
            self._num_estimators,
            expand=self._partition_rows,
        )
        if self._partition_rows:
            x = _split_members(x)
        x = self.recipe.features.transform_ensemble(x)
        if x.num_members != self.num_members:
            raise ValueError(
                "Expected inputs to map to the same number of ensemble members"
            )

        related_ensembles: Mapping[str, EnsembleTable] = {}
        if related_tables is not None:
            assert self._related_processors is not None
            for name, table in related_tables.tables.items():
                processor = self._related_processors[name]
                related_ensembles[name] = processor.transform_ensemble(
                    _to_ensemble_table(table, self._num_estimators)
                )
                if related_ensembles[name].num_members != self.num_members:
                    raise ValueError(
                        "Expected inputs to map to the same number of "
                        "ensemble members"
                    )

        members: list[MemberQuery] = []
        for member_id in range(self.num_members):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = RelatedTables(
                    tables={
                        name: table.table(member_id)
                        for name, table in related_ensembles.items()
                    },
                    relationships=related_tables.relationships,
                    task_links=related_tables.task_links,
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
        assert len(outputs) == self.num_members

        # Reconstruct the group layout of the transformed target:
        assert self._y_locations is not None
        num_groups = max(group for group, _ in self._y_locations) + 1
        groups: list[list[TableTensor | None]] = [
            [] for _ in range(num_groups)
        ]
        for group_id, _ in self._y_locations:
            groups[group_id].append(None)
        for i, (group_id, position) in enumerate(self._y_locations):
            groups[group_id][position] = outputs[i]

        table = EnsembleTable._from_groups(
            groups=[
                cast(
                    TableTensor,
                    group[0].unsqueeze(0)  # type: ignore
                    if len(group) == 1
                    else torch.stack(group, dim=0),  # type: ignore
                )
                for group in groups
            ],
            locations=self._y_locations,
        )

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
            expected = set(outputs[0].columns[Stype.numerical])
            for output in outputs[1:]:
                if set(output.columns[Stype.numerical]) != expected:
                    raise ValueError(
                        "Expected all model outputs to have the same columns "
                        "before applying 'Recipe.output'. Ensure every target "
                        "contains the same set of classes."
                    )

            out = torch.stack(list(outputs), dim=0)

        return self.recipe.output.transform(cast(TableTensor, out))


def _partition_rows(
    x: EnsembleTable,
    y: EnsembleTable,
    *,
    generator: torch.Generator | None,
) -> tuple[EnsembleTable, EnsembleTable]:
    """Randomly partition aligned context rows across ensemble members."""
    x_table = x.table(0)
    if x_table.size(-2) != y.table(0).size(-2):
        raise ValueError(
            f"Expected features and targets to have matching row dimensions "
            f"(got {x_table.size(-2)} and {y.table(0).size(-2)})"
        )
    permutation = torch.randperm(
        x_table.size(-2),
        generator=generator,
        device=x_table.device,
    )
    x_table = cast(TableTensor, x_table.index_select(-2, permutation))

    num_members = x.num_members
    rows_per_member, remainder = divmod(x_table.size(-2), num_members)
    x_groups: list[TableTensor] = []
    y_groups: list[TableTensor] = []
    start = 0
    for member_id in range(num_members):
        num_rows = rows_per_member + (member_id < remainder)
        member_rows = permutation.narrow(0, start, num_rows)
        x_groups.append(
            cast(
                TableTensor,
                x_table.narrow(-2, start, num_rows).unsqueeze(0),
            )
        )
        y_table = y.table(member_id)
        y_groups.append(
            cast(
                TableTensor,
                y_table.index_select(
                    -2,
                    member_rows.to(y_table.device),
                ).unsqueeze(0),
            )
        )
        start += num_rows

    locations = tuple((member_id, 0) for member_id in range(num_members))
    return (
        EnsembleTable._from_groups(x_groups, locations),
        EnsembleTable._from_groups(y_groups, locations),
    )


def _split_members(ensemble_table: EnsembleTable) -> EnsembleTable:
    """Represent every logical member as a distinct processing group."""
    groups = [
        cast(TableTensor, ensemble_table.table(i).unsqueeze(0))
        for i in range(ensemble_table.num_members)
    ]
    locations = tuple((i, 0) for i in range(ensemble_table.num_members))
    return EnsembleTable._from_groups(groups, locations)


def _to_ensemble_table(
    x: Tensor | TableTensor | EnsembleTable,
    num_estimators: int | None,
    *,
    expand: bool = False,
) -> EnsembleTable:
    r"""Convert model input data into an ensemble-aware representation.

    Args:
        x: The input table.
        num_estimators: Number of estimators to represent. If ``None``, a 2D
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
        if x.num_members < 1:
            raise ValueError("'num_estimators' needs to be positive")
        return x

    if not isinstance(x, TableTensor):
        x = TableTensor.from_tensor(x)

    # Treat leading dimension as ensemble dimension:
    if x.dim() > 2 and num_estimators is None:
        locations = tuple((0, i) for i in range(x.size(0)))
        x = EnsembleTable._from_groups((x,), locations)
        if x.num_members < 1:
            raise ValueError("'num_estimators' needs to be positive")
        return x

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
