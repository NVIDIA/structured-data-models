# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from sdm import RelatedTables, Stype, StypeLike, TableTensor
from sdm.processing.execution import MemberContext, MemberQuery
from sdm.tensor.table import TableSchema


def _stack_tables(
    tables: Sequence[TableTensor],
    *,
    target: bool = False,
) -> TableTensor:
    if len(tables) == 1:
        return tables[0]

    ref = tables[0]
    for table in tables[1:]:
        # Numerical feature positions can differ after column shuffling.
        # Other metadata must agree because the batch shares one schema.
        if table.active_stypes != ref.active_stypes or any(
            columns != ref.columns[stype]
            for stype, columns in table.columns.items()
            if target or stype != Stype.numerical
        ):
            raise ValueError(
                "Estimator batches require compatible column layouts; "
                "use 'estimator_batch_size=1' for incompatible estimators"
            )
        for categories, ref_categories in zip(
            table.categorical.categories,
            ref.categorical.categories,
            strict=True,
        ):
            compatible = (
                len(categories) == len(ref_categories)
                if target
                else categories is ref_categories
                or categories.equal(ref_categories)
            )
            if not compatible:
                raise ValueError(
                    "Estimator batches require matching class counts and "
                    "compatible feature categories; use "
                    "'estimator_batch_size=1' for incompatible estimators"
                )

    # TableTensor.stack aligns names, which would undo feature permutations.
    # Stack blocks by position; target class labels are restored on output.
    return TableTensor(
        columns=cast(dict[StypeLike, tuple[str, ...]], ref.columns),
        size=(len(tables), *ref.size()[:-1]),
        device=ref.device,
        **{
            stype: torch.stack(
                [table.blocks[stype] for table in tables], dim=0
            )
            for stype, _ in ref.items()
        },
    )


def _stack_related(
    tables: Sequence[RelatedTables[TableTensor] | None],
) -> RelatedTables[TableTensor] | None:
    ref = tables[0]
    if len(tables) == 1 or ref is None:
        return ref
    if any(table is None or table.schema != ref.schema for table in tables):
        raise ValueError(
            "Estimator batches require compatible related table schemas; "
            "use 'estimator_batch_size=1' for incompatible estimators"
        )
    tables = cast(Sequence[RelatedTables[TableTensor]], tables)
    return ref.replace_tables(
        {
            name: _stack_tables([table.tables[name] for table in tables])
            for name in ref.tables
        }
    )


def _stack_contexts(contexts: Sequence[MemberContext]) -> MemberContext:
    return MemberContext(
        x=_stack_tables([context.x for context in contexts]),
        y=_stack_tables([context.y for context in contexts], target=True),
        related_tables=_stack_related(
            [context.related_tables for context in contexts]
        ),
    )


def _stack_queries(queries: Sequence[MemberQuery]) -> MemberQuery:
    return MemberQuery(
        x=_stack_tables([query.x for query in queries]),
        related_tables=_stack_related(
            [query.related_tables for query in queries]
        ),
    )


def _output_columns(
    contexts: Sequence[MemberContext],
) -> tuple[tuple[str, ...], ...] | None:
    if len(contexts) == 1 or contexts[0].y.categorical.size(-1) == 0:
        return None
    return tuple(
        tuple(
            str(value)
            for value in context.y.categorical.categories[0].tolist()
        )
        for context in contexts
    )


def _unstack_output(
    out: TableTensor,
    num_members: int,
    columns: Sequence[tuple[str, ...]] | None,
) -> list[TableTensor]:
    outputs = (
        [out]
        if num_members == 1
        else list(cast(tuple[TableTensor, ...], out.unbind(0)))
    )
    if columns is None:
        return outputs
    return [
        TableTensor(
            columns={Stype.numerical: names},
            numerical=output.numerical,
        )
        for output, names in zip(outputs, columns, strict=True)
    ]


def _categorical_mask(
    x: TableTensor,
    schema: TableSchema,
    schemas: Sequence[TableSchema],
) -> Tensor:
    categorical_columns = set(schema.columns[Stype.categorical])
    mask = torch.tensor(
        [
            [
                column in categorical_columns
                for column in s.columns[Stype.numerical]
            ]
            for s in schemas
        ],
        device=x.device,
        dtype=torch.bool,
    )
    if len(schemas) == 1:
        return mask[0]
    # [E, C] -> [E, 1, ..., C], preserving existing input batch dimensions.
    return mask.view(len(schemas), *((1,) * (x.dim() - 3)), -1)
