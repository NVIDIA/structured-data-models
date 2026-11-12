# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import NamedTuple, cast, overload

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import (
    EnsembleData,
    EnsembleTable,
    Recipe,
    RelatedTables,
    Stype,
    TableLike,
    TableTensor,
)
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.recipe import _iter_modules


class MemberContext(NamedTuple):
    """Transformed context tables for one ensemble member."""

    x: TableTensor
    y: TableTensor
    x_context_only: TableTensor | None
    related_tables: RelatedTables[TableTensor] | None
    #: Semantic types of the raw context columns before ``recipe.features``.
    input_stypes: Mapping[str, Stype]


class MemberQuery(NamedTuple):
    """Transformed query tables for one ensemble member."""

    x: TableTensor
    related_tables: RelatedTables[TableTensor] | None


class RecipeExecution:
    """Recipe execution manager during model processing."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

        self._related_processors: Mapping[str, Processor] | None = None
        self._num_estimators: int | None = None
        self._num_members: int | None = None

    @property
    def num_members(self) -> int:
        r"""The number of fitted members."""
        assert self._num_members is not None
        return self._num_members

    @overload
    def fit_transform(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        related_tables: RelatedTables | None,
        *,
        num_members: int | None = None,
        context_only_columns: Sequence[str] | Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[MemberContext, ...]: ...

    @overload
    def fit_transform(
        self,
        x: Tensor | TableTensor,
        y: None,
        related_tables: RelatedTables | None,
        *,
        num_members: int | None = None,
        context_only_columns: None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[MemberQuery, ...]: ...

    def fit_transform(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor | None,
        related_tables: RelatedTables | None,
        *,
        num_members: int | None = None,
        context_only_columns: Sequence[str] | Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[MemberContext | MemberQuery, ...]:
        """Fit and transform context data."""
        self._num_estimators = num_members

        # Transform the target first to resolve the task type.
        series_processors: tuple[sp.SeriesProcessor, ...] = ()
        if y is not None:
            y: TableLike = _to_table(y, num_members)

            for processor in (
                self.recipe.features,
                self.recipe.output,
            ):
                for module in _iter_modules(processor):
                    if isinstance(module, sp.TargetAwareProcessor):
                        module._target = y

            series_processors = tuple(
                module
                for processor in (self.recipe.features, self.recipe.target)
                for module in _iter_modules(processor)
                if isinstance(module, sp.SeriesProcessor)
            )
            for module in series_processors:
                module._set_buffer(
                    name="_offset",
                    value=torch.zeros((), dtype=torch.long, device=y.device),
                )

            y = self.recipe.target.fit_transform(y, generator=generator)
            targets = y if isinstance(y, EnsembleTable) else (y,)
            self._num_members = len(targets)

            task_dispatchers = tuple(
                module
                for processor in (self.recipe.features, self.recipe.output)
                for module in _iter_modules(processor)
                if isinstance(module, sp.TaskDispatch)
            )
            if len(task_dispatchers) > 0:
                if any(target.size(-1) != 1 for target in targets):
                    raise ValueError(
                        "Expected the transformed target to contain exactly "
                        "one column"
                    )

                if all(target.numerical.size(-1) == 1 for target in targets):
                    task = "regression"
                elif all(
                    target.categorical.size(-1) == 1 for target in targets
                ):
                    task = "classification"
                else:
                    raise ValueError(
                        "'Recipe.target' must resolve to a single task type"
                    )

                for task_dispatcher in task_dispatchers:
                    task_dispatcher._task = task

        self._related_processors = None
        related_outputs: Mapping[str, TableLike] = {}
        if related_tables is not None:
            self._related_processors = {}
            for name, table in related_tables.tables.items():
                processor = copy.deepcopy(self.recipe.features)
                for module in _iter_modules(processor):
                    if isinstance(module, sp.TableDispatch):
                        module._route = "related"
                self._related_processors[name] = processor
                related_input = _to_table(table, num_members)
                related_outputs[name] = processor.fit_transform(
                    related_input,
                    generator=generator,
                )
                for module in _iter_modules(processor):
                    if isinstance(module, sp.SeriesProcessor):
                        module._set_buffer(
                            name="_offset",
                            value=EnsembleData.execute(
                                torch.scalar_tensor,
                                related_input.size(-2),
                                dtype=torch.long,
                                device=related_input.device,
                            ),
                        )

        for module in _iter_modules(self.recipe.features):
            if isinstance(module, sp.TableDispatch):
                module._route = "task"

        x_context_only: TableLike | None = None
        if context_only_columns:
            assert y is not None
            if not isinstance(x, TableTensor):
                x = TableTensor.from_tensor(cast(Tensor, x))
            context_only_columns = [str(col) for col in context_only_columns]
            processor = copy.deepcopy(self.recipe.features)
            x_context_only = processor.fit_transform(
                _to_table(x.select_columns(context_only_columns), num_members),
                generator=generator,
            )
            x = x.drop_columns(context_only_columns)

        inputs = _to_table(x, num_members)
        x: TableLike = self.recipe.features.fit_transform(
            inputs,
            generator=generator,
        )
        features = x if isinstance(x, EnsembleTable) else (x,)
        self._num_members = len(features)

        for module in series_processors:
            module._set_buffer(
                name="_offset",
                value=EnsembleData.execute(
                    torch.scalar_tensor,
                    x.size(-2),
                    dtype=torch.long,
                    device=x.device,
                ),
            )

        members: list[MemberContext | MemberQuery] = []
        for member_id, features_i in enumerate(features):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = RelatedTables(
                    tables={
                        name: table[member_id]
                        if isinstance(table, EnsembleTable)
                        else table
                        for name, table in related_outputs.items()
                    },
                    relationships=related_tables.relationships,
                    task_links=related_tables.task_links,
                )
            if y is not None:
                member = MemberContext(
                    x=features_i,
                    y=y[member_id] if isinstance(y, EnsembleTable) else y,
                    x_context_only=x_context_only[member_id]
                    if isinstance(x_context_only, EnsembleTable)
                    else x_context_only,
                    related_tables=related_tables_i,
                    input_stypes=(
                        inputs[member_id]
                        if isinstance(inputs, EnsembleTable)
                        else inputs
                    ).stypes,
                )
            else:
                member = MemberQuery(
                    x=features_i,
                    related_tables=related_tables_i,
                )
            members.append(member)

        return tuple(members)

    def transform(
        self,
        x: Tensor | TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[MemberQuery, ...]:
        """Transform query data."""
        x: TableLike = _to_table(x, self._num_estimators)
        x = self.recipe.features.transform(x)

        related_outputs: Mapping[str, TableLike] = {}
        if related_tables is not None:
            assert self._related_processors is not None
            for name, table in related_tables.tables.items():
                processor = self._related_processors[name]
                related_input = _to_table(table, self._num_estimators)
                related_outputs[name] = processor.transform(related_input)

        members: list[MemberQuery] = []
        features = x if isinstance(x, EnsembleTable) else (x,)
        for member_id, features_i in enumerate(features):
            related_tables_i: RelatedTables[TableTensor] | None = None
            if related_tables is not None:
                related_tables_i = RelatedTables(
                    tables={
                        name: table[member_id]
                        if isinstance(table, EnsembleTable)
                        else table
                        for name, table in related_outputs.items()
                    },
                    relationships=related_tables.relationships,
                    task_links=related_tables.task_links,
                )
            members.append(
                MemberQuery(
                    x=features_i,
                    related_tables=related_tables_i,
                )
            )

        return tuple(members)

    def inverse_transform_target(
        self, outputs: Sequence[TableTensor]
    ) -> tuple[TableTensor, ...]:
        """Invert fitted target transforms on corresponding member outputs."""
        table = (
            outputs[0]
            if len(outputs) == 1
            else EnsembleTable.from_tables(
                outputs, member_table_ids=range(len(outputs))
            )
        )
        if not isinstance(self.recipe.target, InvertibleMixin):
            raise RuntimeError("Target recipe is not invertible")
        table = self.recipe.target.inverse_transform(table)
        return tuple(table) if isinstance(table, EnsembleTable) else (table,)

    def transform_output(
        self,
        outputs: Sequence[TableTensor],
    ) -> TableTensor:
        """Apply ``recipe.output`` to member outputs."""
        if len(outputs) == 1:
            out = outputs[0].unsqueeze(0)
        else:
            expected = set(outputs[0].numerical_columns)
            for output in outputs[1:]:
                if set(output.numerical_columns) != expected:
                    raise ValueError(
                        "Expected all model outputs to have the same columns "
                        "before applying 'Recipe.output'. Ensure every target "
                        "contains the same set of classes."
                    )

            out = torch.stack(list(outputs), dim=0)

        return self.recipe.output.transform(cast(TableTensor, out))


def _to_table(
    x: Tensor | TableLike,
    num_estimators: int | None,
) -> TableLike:
    r"""Normalize model input, creating an ensemble only for multiple members.

    Args:
        x: The input table.
        num_estimators: Number of estimators to represent. If ``None``, a 2D
            input creates one estimator and higher-rank inputs infer the
            estimator count from their leading dimension.
    """
    if isinstance(x, EnsembleTable):
        if num_estimators is not None and num_estimators != len(x):
            raise ValueError(
                f"Expected {num_estimators} members in 'EnsembleTable' "
                f"(got {len(x)})"
            )
        if len(x) < 1:
            raise ValueError("'num_estimators' needs to be positive")
        return x

    if not isinstance(x, TableTensor):
        x = TableTensor.from_tensor(x)

    # Treat leading dimension as ensemble dimension:
    if x.dim() > 2 and num_estimators is None:
        if x.size(0) == 1:
            return x[0]
        locations = tuple((0, i) for i in range(x.size(0)))
        x = EnsembleTable(groups=(x,), locations=locations)
        if len(x) < 1:
            raise ValueError("'num_estimators' needs to be positive")
        return x

    num_estimators = 1 if num_estimators is None else num_estimators
    if num_estimators < 1:
        raise ValueError("'num_estimators' needs to be positive")

    if num_estimators == 1:
        return x
    return EnsembleTable.from_table(x, num_members=num_estimators)
