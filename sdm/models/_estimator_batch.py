# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from sdm.processing.execution import MemberContext, MemberQuery

T = TypeVar("T")


@dataclass(frozen=True)
class EstimatorBatch:
    r"""Logical ensemble members evaluated in one model call."""

    member_ids: tuple[int, ...]

    def select(self, members: Sequence[T]) -> tuple[T, ...]:
        r"""Select this batch's members from an ensemble-ordered sequence."""
        return tuple(members[i] for i in self.member_ids)


def plan_estimator_batches(
    contexts: Sequence[MemberContext],
    queries: Sequence[MemberQuery] | None,
    class_values: Sequence[tuple[Any, ...] | None],
    estimator_batch_size: int | None,
) -> tuple[EstimatorBatch, ...]:
    r"""Plan consecutive compatible members into model calls."""
    if len(contexts) == 0:
        return ()

    related = any(context.related_tables is not None for context in contexts)
    if queries is not None:
        related = related or any(
            query.related_tables is not None for query in queries
        )
    if related:
        return tuple(
            EstimatorBatch(member_ids=(i,)) for i in range(len(contexts))
        )

    batches: list[EstimatorBatch] = []
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
            tuple(
                (
                    tuple(
                        (stype, block.size(), block.dtype)
                        for stype, block in table.items()
                    ),
                    tuple(c.numel() for c in table.categorical.categories),
                )
                for table in tables
            ),
            None if classes is None else frozenset(classes),
        )
        if i > start and (
            member_key != key
            or (
                estimator_batch_size is not None
                and i - start == estimator_batch_size
            )
        ):
            batches.append(
                EstimatorBatch(member_ids=tuple(range(start, i)))
            )
            start = i
        key = member_key

    batches.append(
        EstimatorBatch(member_ids=tuple(range(start, len(contexts))))
    )
    return tuple(batches)
