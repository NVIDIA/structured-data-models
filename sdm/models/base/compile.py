# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Compiled components of :meth:`~sdm.models.ICLModel.predict`.

A compiled predict runs the steps that organize the work as plain Python:
validation, the loop over estimator batches, the prefetch of the fitted cache
and the stacking of estimators. It compiles only components, functions that
do the work on tensors, *e.g.*, the model of one estimator batch.
:meth:`~sdm.models.ICLModel._predict` takes the components in place of its
default steps, so that each model chooses its own.

A graph of a component specializes on the Python values among its inputs,
*e.g.*, names of columns, and on sizes of zero and one. Keep both outside of
components to reuse their graphs after fits on other data.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import EnsembleTable, RelatedTables, Stype, TableTensor
from sdm.processing.base import TensorStep
from sdm.processing.common.sequential import _chain_tensor_steps

if TYPE_CHECKING:
    from sdm.models.base import Callback


def compile_component(
    fn: Callable[..., Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    r"""Compile a component of predict using :func:`torch.compile`.

    Args:
        fn: The component.
        kwargs: Keyword arguments passed to :func:`torch.compile`.
            ``dynamic`` defaults to ``True``, so that all sizes are dynamic
            from the first call; with ``None``, sizes are static until they
            change or are marked, *e.g.*, by
            :func:`torch._dynamo.maybe_mark_dynamic`. ``isolate_recompiles``
            defaults to ``True``, since all components would otherwise share
            one recompile limit.
    """
    kwargs = {"dynamic": True, "isolate_recompiles": True} | kwargs
    # A copy, from which `torch.compile` removes some options.
    if kwargs.get("options") is not None:
        kwargs["options"] = dict(kwargs["options"])
    # Sizes that are equal in the first call would otherwise share one symbol,
    # and compile another graph once they differ.
    return torch.fx.experimental._config.patch(use_duck_shape=False)(
        torch.compile(fn, **kwargs)
    )


def repeat_single_rows(
    predict: Callable[..., TableTensor],
    uncompiled: Callable[..., TableTensor],
) -> Callable[..., TableTensor]:
    r"""Keep queries of zero and one rows away from compiled components.

    A query without rows runs ``uncompiled``, and a query of one row is
    repeated and its first row returned, unless callbacks or related tables
    see the rows.

    Args:
        predict: The compiled predict.
        uncompiled: The uncompiled predict.
    """

    def wrapper(
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> TableTensor:
        rows = x.size(-2)
        if rows == 0:
            return uncompiled(x, related_tables, callbacks=callbacks)
        if rows > 1 or related_tables is not None or callbacks:
            return predict(x, related_tables, callbacks=callbacks)
        out = predict(torch.cat([x, x], dim=-2), callbacks=callbacks)
        return cast(TableTensor, out[..., :1, :])

    return wrapper


class _Route(NamedTuple):
    # The columns of one stype, which `head` transforms as tables, *e.g.*,
    # looking up categories, before tensor steps transform their blocks.
    stype: Stype
    head: tuple[sp.EnsembleProcessor, ...]
    locations: tuple[tuple[int, int], ...]


class _Member(NamedTuple):
    # Per route, the group and position of the member's block and its tensor
    # steps; then the tensor steps of the concatenated blocks.
    routes: tuple[tuple[int, int, tuple[TensorStep, ...]], ...]
    steps: tuple[TensorStep, ...]


class _FeaturePlan(NamedTuple):
    # The features of a fit as tensor steps: `eager` and the heads of the
    # routes of `dispatch` run as tables, the tensor steps of all members in
    # one call.
    columns: tuple[tuple[Stype, tuple[str, ...]], ...]
    eager: tuple[sp.EnsembleProcessor, ...]
    routes: tuple[_Route, ...]
    members: tuple[_Member, ...]
    member_columns: tuple[tuple[str, ...], ...]


_plans: weakref.WeakKeyDictionary[sp.EnsembleProcessor, _FeaturePlan | None]
_plans = weakref.WeakKeyDictionary()


def transform_features(
    members: Callable[..., list[Tensor]],
    processor: sp.EnsembleProcessor,
    ensemble_table: EnsembleTable,
) -> EnsembleTable:
    r"""Transform features by a recipe, its tensor work in one call.

    The processors of a fit become tensor steps, functions of the blocks of
    each member and of fitted tensors, see
    :meth:`~sdm.processing.EnsembleProcessor._member_tensor_steps`.
    ``members``, *e.g.*, :func:`transform_members` compiled, runs the tensor
    steps after the routes of all members, so that its graphs see only
    tensors, whose sizes are dynamic, and the functions and settings of the
    steps, and serve fits on tables of other columns. The tensor steps of
    the routes run uncompiled. Processors without tensor steps, *e.g.*, the
    lookup of categories, and the names of columns stay plain Python. A
    recipe of other shape runs uncompiled.

    Args:
        members: Runs the tensor steps, called like
            :func:`transform_members`.
        processor: The fitted features of the recipe.
        ensemble_table: The query.
    """
    plan = _plans.get(processor, False)
    if plan is not None and (
        plan is False or plan.columns != _columns(ensemble_table)
    ):
        plan = _plans[processor] = _feature_plan(processor, ensemble_table)
    if plan is None:
        return processor.transform_ensemble(ensemble_table)

    for child in plan.eager:
        ensemble_table = child.transform_ensemble(ensemble_table)
    inputs = []
    for route in plan.routes:
        table = ensemble_table.select_stypes(route.stype)
        for child in route.head:
            table = child.transform_ensemble(table)
        assert table._locations == route.locations
        inputs.append([group.numerical for group in table._iter_groups()])
    # The routes run uncompiled, as which routes run, and sizes of one, on
    # which a graph specializes, differ between fits.
    outs = members(
        _route_blocks(inputs, plan.members),
        [member.steps for member in plan.members],
    )
    return EnsembleTable.from_tables(
        tables=[
            TableTensor(columns={"numerical": columns}, numerical=out)
            for columns, out in zip(plan.member_columns, outs, strict=True)
        ],
        member_table_ids=range(len(outs)),
    )


def transform_members(
    blocks: Sequence[Tensor],  # per member [1, R, C]
    steps: Sequence[Sequence[TensorStep]],  # per member
) -> list[Tensor]:  # per member [R, C]
    r"""Run the tensor steps of each member on its block."""
    outs = []
    for x, member_steps in zip(blocks, steps, strict=True):
        for step in member_steps:
            x = step.fn(x, *step.args)
        outs.append(x[0])
    return outs


def _route_blocks(
    inputs: Sequence[Sequence[Tensor]],  # per route, per group [B, R, C]
    members: Sequence[_Member],
) -> list[Tensor]:  # per member [1, R, C]
    blocks = []
    for member in members:
        routes = []
        for groups, (group_id, position, steps) in zip(
            inputs, member.routes, strict=True
        ):
            x = groups[group_id][position : position + 1]
            # TODO: Split the rows, as the transforms of `RankGaussian` and
            # `ClipSigma` do, to bound the memory of large queries.
            for step in steps:
                x = step.fn(x, *step.args)
            routes.append(x)
        # A copy also of a single route, as a graph specializes on views.
        blocks.append(torch.cat(routes, dim=-1))
    return blocks


def _columns(
    ensemble_table: EnsembleTable,
) -> tuple[tuple[Stype, tuple[str, ...]], ...]:
    return tuple(
        tuple(group.column_groups) for group in ensemble_table._iter_groups()
    )


def _feature_plan(
    processor: sp.EnsembleProcessor,
    ensemble_table: EnsembleTable,
) -> _FeaturePlan | None:
    # Supported: `Sequential(*eager, StypeDispatch, *tail)`, whose tail has
    # tensor steps.
    children = list(processor) if isinstance(processor, sp.Sequential) else []
    cut = next(
        (
            i
            for i, child in enumerate(children)
            if isinstance(child, sp.StypeDispatch)
        ),
        None,
    )
    if cut is None:
        return None
    schema = _columns(ensemble_table)
    dispatch = cast(sp.StypeDispatch, children[cut])
    eager, tail = tuple(children[:cut]), children[cut + 1 :]
    for child in eager:
        ensemble_table = child.transform_ensemble(ensemble_table)
    active = dispatch._active_routes
    if active is None or any(
        names and str(s) not in active
        for group in ensemble_table._iter_groups()
        for s, names in group.column_groups
    ):
        return None
    num_members = len(ensemble_table)

    routes = []
    route_steps: list[list[tuple[int, int, tuple[TensorStep, ...]]]]
    route_steps = [[] for _ in range(num_members)]
    member_columns: list[tuple[str, ...]] = [() for _ in range(num_members)]
    for stype in active:
        route = dispatch.processors[stype]
        processors = (
            list(route) if isinstance(route, sp.Sequential) else [route]
        )
        table = ensemble_table.select_stypes(stype)
        # The leading processors of a route without tensor steps run as
        # tables.
        head = []
        while True:
            found = None
            if all(
                not names or s == Stype.numerical
                for group in table._iter_groups()
                for s, names in group.column_groups
            ):
                found = [
                    _chain_tensor_steps(
                        processors, i, table[i].numerical_columns
                    )
                    for i in range(num_members)
                ]
            if found is not None and all(out is not None for out in found):
                break
            if not processors:
                return None
            head.append(processors.pop(0))
            table = head[-1].transform_ensemble(table)
        routes.append(_Route(Stype(stype), tuple(head), table._locations))
        for i, (group_id, position) in enumerate(table._locations):
            steps, columns = cast(tuple, found[i])
            route_steps[i].append((group_id, position, steps))
            member_columns[i] += columns

    members = []
    for i in range(num_members):
        found_tail = _chain_tensor_steps(tail, i, member_columns[i])
        if found_tail is None:
            return None
        members.append(_Member(tuple(route_steps[i]), found_tail[0]))
        member_columns[i] = found_tail[1]
    return _FeaturePlan(
        columns=schema,
        eager=eager,
        routes=tuple(routes),
        members=tuple(members),
        member_columns=tuple(member_columns),
    )
