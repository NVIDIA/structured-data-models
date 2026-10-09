# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor

from sdm import Stype, StypeLike, TableTensor
from sdm.processing.execution import MemberContext, MemberQuery


def _table_layout(table: TableTensor) -> Hashable:
    r"""Describe tensor shapes, dtypes, and categorical vocabulary sizes."""
    return (
        tuple(
            (stype, block.size(), block.dtype)
            for stype, block in table.items()
        ),
        tuple(c.numel() for c in table.categorical.categories),
    )


def _batch_slices(
    contexts: Sequence[MemberContext],
    queries: Sequence[MemberQuery] | None,
    class_values: Sequence[tuple[Any, ...] | None],
    estimator_batch_size: int | None,
) -> list[slice]:
    r"""Group consecutive compatible estimators into model calls."""
    if estimator_batch_size == 1:
        return [slice(i, i + 1) for i in range(len(contexts))]

    related = any(context.related_tables is not None for context in contexts)
    if queries is not None:
        related = related or any(
            query.related_tables is not None for query in queries
        )
    if related:
        return [slice(i, i + 1) for i in range(len(contexts))]

    batches: list[slice] = []
    start = 0
    key: Hashable = None
    for i, context in enumerate(contexts):
        query = None if queries is None else queries[i]
        tables = (
            [context.x, context.y]
            if query is None
            else [context.x, context.y, query.x]
        )
        classes = class_values[i]
        member_key = (
            tuple(_table_layout(table) for table in tables),
            None if classes is None else frozenset(classes),
        )
        if i > start and (
            member_key != key
            or (
                estimator_batch_size is not None
                and i - start == estimator_batch_size
            )
        ):
            batches.append(slice(start, i))
            start = i
        key = member_key

    batches.append(slice(start, len(contexts)))
    return batches


def _stack(tables: Sequence[TableTensor]) -> TableTensor:
    ref = tables[0]
    if len(tables) == 1:
        return ref
    # torch.stack aligns columns by name, which would undo per-estimator column
    # shuffles; renaming to the first member's names stacks blocks by position.
    columns = cast(Mapping[StypeLike, Sequence[str]], ref.columns)
    renamed: list[Tensor] = [
        ref,
        *(
            table.__class__(columns=columns, **dict(table.items()))
            for table in tables[1:]
        ),
    ]
    return cast(TableTensor, torch.stack(renamed))


def _stack_context(members: Sequence[MemberContext]) -> MemberContext:
    if len(members) == 1:
        return members[0]
    return MemberContext(
        x=_stack([member.x for member in members]),
        y=_stack([member.y for member in members]),
        related_tables=None,
        input_stypes=members[0].input_stypes,
    )


def _stack_query(members: Sequence[MemberQuery]) -> MemberQuery:
    if len(members) == 1:
        return members[0]
    return MemberQuery(
        x=_stack([member.x for member in members]), related_tables=None
    )


def _categorical_mask(members: Sequence[MemberContext]) -> Tensor:
    x = members[0].x
    mask = torch.tensor(
        [
            [
                member.input_stypes.get(column) == Stype.categorical
                for column in member.x.columns[Stype.numerical]
            ]
            for member in members
        ],
        dtype=torch.bool,
        device=x.device,
    )  # [E, C]
    if len(members) == 1:
        return mask[0]
    # Insert the member's batch dimensions so the mask broadcasts over them:
    return mask.view(len(members), *(1,) * (x.dim() - 2), -1)  # [E, 1, ..., C]


def _class_values(
    contexts: Sequence[MemberContext],
    estimator_batch_size: int | None,
) -> list[tuple[Any, ...] | None]:
    # Class labels per estimator, used to group and relabel stacked batches.
    # Unused when each estimator already has its own ``_forward``.
    if (
        estimator_batch_size == 1
        or contexts[0].related_tables is not None
        or contexts[0].y.categorical.size(-1) == 0
    ):
        return [None] * len(contexts)
    return [
        tuple(context.y.categorical.categories[0].tolist())
        for context in contexts
    ]


def _unstack(
    out: TableTensor,
    class_values: Sequence[tuple[Any, ...] | None] | None,
    num_members: int,
) -> list[TableTensor]:
    outs = (
        [out]
        if num_members == 1
        else list(cast(tuple[TableTensor, ...], out.unbind(0)))
    )
    if class_values is None or len(class_values) == 1:
        return outs
    first = class_values[0]
    if first is None:
        return outs
    # The model labels columns in the first member's class order; member `e`'s
    # column `j` holds class `class_values[e][j]`.
    labels = out.columns[Stype.numerical]
    index = {value: i for i, value in enumerate(first)}
    stacked = cast(Sequence[tuple[Any, ...]], class_values)
    return [
        TableTensor(
            columns={Stype.numerical: [labels[index[v]] for v in classes]},
            numerical=member_out.numerical,
        )
        for member_out, classes in zip(outs, stacked, strict=True)
    ]
