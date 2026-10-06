# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Hashable, Sequence
from typing import Any

from sdm.processing.execution import MemberContext, MemberQuery


def _batch_slices(
    contexts: Sequence[MemberContext],
    queries: Sequence[MemberQuery] | None,
    class_values: Sequence[tuple[Any, ...] | None],
    estimator_batch_size: int | None,
) -> list[slice]:
    r"""Group consecutive compatible estimators into model calls."""
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
            batches.append(slice(start, i))
            start = i
        key = member_key

    batches.append(slice(start, len(contexts)))
    return batches
