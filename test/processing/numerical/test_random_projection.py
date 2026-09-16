# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm import CategoricalTensor, EnsembleTable, TableTensor
from sdm.processing import RandomProjection


def _table() -> TableTensor:
    return TableTensor(
        numerical=torch.randn(6, 4),
        categorical=CategoricalTensor(
            torch.randint(0, 2, (6, 1)), categories=(torch.arange(2),)
        ),
    )


def test_random_projection() -> None:
    table = _table()

    inp = EnsembleTable.from_table(table, num_members=8)
    out = RandomProjection(8).fit_transform_ensemble(inp)

    assert out.num_groups == 1
    assert out.num_members == 8
    group = out._groups[0]
    assert group.size() == (8, 6, 9)
    assert group.numerical.size() == (8, 6, 8)
    assert group.numerical.stride() == (6 * 8, 8, 1)
    assert group.categorical.size() == (8, 6, 1)
    assert group.categorical.stride() == (0, 1, 1)


def _ensemble(table: TableTensor, layout: str) -> EnsembleTable:
    if layout == "shared":
        return EnsembleTable.from_table(table, num_members=3)
    if layout in {"stacked", "reordered"}:
        order = (0, 1, 2) if layout == "stacked" else (2, 0, 1)
        return EnsembleTable(
            groups=(cast(TableTensor, torch.stack([table] * 3)),),
            locations=tuple((0, i) for i in order),
        )
    if layout == "shared_split":
        return EnsembleTable.from_tables(
            tables=(table, cast(TableTensor, table.clone())),
            member_table_ids=(1, 0, 1),
        )
    return EnsembleTable.from_tables(
        tables=(
            table,
            cast(TableTensor, table.clone()),
            cast(TableTensor, table.clone()),
        ),
        member_table_ids=(2, 0, 1),
    )


@pytest.mark.parametrize(
    ("fitted_layout", "query_layout", "restore"),
    [
        ("shared", "stacked", False),
        ("stacked", "shared", False),
        ("stacked", "reordered", False),
        ("shared", "split", False),
        ("shared_split", "shared", False),
        ("shared_split", "shared", True),
    ],
)
def test_projection_follows_members_across_group_layouts(
    fitted_layout: str, query_layout: str, restore: bool
) -> None:
    table = _table()
    context = _ensemble(table, fitted_layout)
    processor = RandomProjection(8).fit_ensemble(context)
    expected = processor.transform_ensemble(context)
    if restore:
        restored = RandomProjection(8)
        restored.load_state_dict(processor.state_dict())
        processor = restored

    output = processor.transform_ensemble(_ensemble(table, query_layout))

    for member in range(3):
        torch.testing.assert_close(
            output.table(member).numerical, expected.table(member).numerical
        )
        assert output.table(member).categorical.equal(table.categorical)


@pytest.mark.parametrize("num_members", [1, 4])
def test_projection_rejects_changed_member_count(num_members: int) -> None:
    table = _table()
    processor = RandomProjection(8).fit_ensemble(_ensemble(table, "shared"))

    with pytest.raises(RuntimeError, match="fitted with 3 ensemble members"):
        processor.transform_ensemble(
            EnsembleTable.from_table(table, num_members=num_members)
        )


def test_projection_preserves_grouped_random_draws() -> None:
    table = _table()
    generator = torch.Generator()
    reference = torch.Generator().set_state(generator.get_state())
    weight = torch.empty(3, 8, 4).normal_(std=8**-0.5, generator=reference)
    processor = RandomProjection(8)

    output = processor.fit_transform_ensemble(
        _ensemble(table, "shared"), generator=generator
    )

    assert torch.equal(generator.get_state(), reference.get_state())
    torch.testing.assert_close(
        torch.stack([output.table(i).numerical for i in range(3)]),
        table.numerical @ weight.transpose(-1, -2),
    )


def test_projection_preserves_unreferenced_groups() -> None:
    table = _table()
    context = EnsembleTable(
        groups=(
            cast(TableTensor, table.unsqueeze(0)),
            cast(TableTensor, table.unsqueeze(0)),
        ),
        locations=((1, 0),),
    )
    processor = RandomProjection(8)
    expected = processor.fit_transform_ensemble(context)
    query = EnsembleTable(
        groups=(cast(TableTensor, table.unsqueeze(0)),) * 3,
        locations=((0, 0),),
    )

    output = processor.transform_ensemble(query)

    assert expected.num_members == output.num_members == 1
    assert [group.size(0) for group in expected] == [0, 1]
    assert [group.size(0) for group in output] == [1, 0, 0]
    torch.testing.assert_close(
        output.table(0).numerical, expected.table(0).numerical
    )
    assert output.table(0).categorical.equal(table.categorical)
