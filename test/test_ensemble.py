# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm import EnsembleTable, EnsembleTensor, TableTensor


def test_tensor_member() -> None:
    group = torch.tensor([[1.0], [2.0]])
    ensemble = EnsembleTensor(
        groups=(group,),
        locations=((0, 1), (0, 0), (0, 1)),
    )

    torch.testing.assert_close(ensemble[0], group[1])


def test_shared_member_table() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_table = EnsembleTable.from_table(data, num_members=3)

    assert ensemble_table.num_members == 3
    assert ensemble_table.num_groups == 1
    assert repr(ensemble_table) == (
        "EnsembleTable(num_members=3, num_groups=1)"
    )
    groups = tuple(ensemble_table)
    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert next(iter(ensemble_table)) is groups[0]
    for member_id in range(3):
        assert ensemble_table[member_id].equal(data)


def test_from_tables_keeps_tables_separate() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    groups = tuple(ensemble_table)
    assert len(groups) == 2
    assert all(group.size() == (1, 2, 1) for group in groups)
    assert ensemble_table.member(0).equal(first)
    assert ensemble_table.member(1).equal(second)
    assert ensemble_table.member(2).equal(first)
    assert ensemble_table.member(3).equal(second)


def test_from_tables_ignores_unused_tables() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0,),
    )

    groups = tuple(ensemble_table)

    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert ensemble_table.member(0).equal(first)


def test_replace_groups_keeps_member_assignment() -> None:
    first = TableTensor.from_tensor(
        tensor=torch.tensor([[1.0], [2.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        tensor=torch.tensor([[3.0], [4.0]]),
        columns=("second",),
    )
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )

    replaced = ensemble_table.replace_groups(
        [
            group.replace_blocks(numerical=-group.numerical)
            for group in ensemble_table
        ]
    )

    assert replaced.num_members == 3
    assert replaced.num_groups == ensemble_table.num_groups
    assert replaced.member(0).numerical.tolist() == [[-3.0], [-4.0]]
    assert replaced.member(1).numerical.tolist() == [[-1.0], [-2.0]]
    assert replaced.member(2).equal(replaced.member(0))
    assert ensemble_table.member(0).equal(second)


def test_replace_groups_rejects_group_count_mismatch() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_table = EnsembleTable.from_table(data, num_members=2)

    with pytest.raises(ValueError, match="one replacement per group"):
        ensemble_table.replace_groups(tuple(ensemble_table) * 2)


def test_replace_tables_packs_only_within_existing_groups() -> None:
    ensemble_table = EnsembleTable(
        groups=(
            TableTensor.from_tensor(torch.tensor([[[0.0]], [[1.0]]])),
            TableTensor.from_tensor(torch.tensor([[[2.0]], [[3.0]]])),
        ),
        locations=((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    tables = tuple(
        TableTensor.from_tensor(torch.tensor([[float(value)]]))
        for value in range(4)
    )

    output = ensemble_table.replace_tables(
        tables=tables,
        member_table_ids=range(4),
    )

    assert output.num_groups == 2
    assert all(group.size(0) == 2 for group in output)
    for member_id, table in enumerate(tables):
        assert output.member(member_id).equal(table)


def test_gather_members_preserves_member_order_and_sharing() -> None:
    tables = tuple(
        TableTensor.from_tensor(torch.tensor([[value]], dtype=torch.float32))
        for value in range(3)
    )
    destination = EnsembleTable.from_table(tables[0], num_members=3)
    first = EnsembleTable(
        groups=(cast(TableTensor, torch.stack(tables[:2])),),
        locations=((0, 0), (0, 1)),
    )
    second = EnsembleTable.from_tables(tables[2:], member_table_ids=(0,))

    output = destination.gather_members(
        tables=(first, second, first),
        member_ids=(1, 0, 1),
    )

    assert output.member(0).equal(tables[1])
    assert output.member(1).equal(tables[2])
    assert output.member(2).equal(tables[1])
    assert next(iter(output)).size(0) == 2


def test_gather_members_rejects_source_count_mismatch() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable.from_table(table, num_members=2)

    with pytest.raises(ValueError, match="one source member"):
        ensemble_table.gather_members(
            tables=(ensemble_table,),
            member_ids=(0, 1),
        )


def test_select_members_preserves_groups_and_order() -> None:
    tables = tuple(
        TableTensor.from_tensor(torch.tensor([[value]], dtype=torch.float32))
        for value in range(3)
    )
    ensemble_table = EnsembleTable(
        groups=(cast(TableTensor, torch.stack(tables)),),
        locations=((0, 2), (0, 0), (0, 1), (0, 2)),
    )

    output = ensemble_table.select_members((1, 3, 0))

    assert output.num_members == 3
    assert output.num_groups == 1
    assert next(iter(output)).size(0) == 2
    assert output.member(0).equal(tables[0])
    assert output.member(1).equal(tables[2])
    assert output.member(2).equal(tables[2])


def test_concatenate_columns_preserves_member_order() -> None:
    left = EnsembleTable.from_tables(
        tables=(
            TableTensor.from_tensor(
                torch.tensor([[1.0], [2.0]]),
                columns=("left",),
            ),
            TableTensor.from_tensor(
                torch.tensor([[3.0], [4.0]]),
                columns=("left",),
            ),
        ),
        member_table_ids=(1, 0, 1),
    )
    right = EnsembleTable.from_tables(
        tables=(
            TableTensor.from_tensor(
                torch.tensor([[10.0], [20.0]]),
                columns=("right",),
            ),
            TableTensor.from_tensor(
                torch.tensor([[30.0], [40.0]]),
                columns=("right",),
            ),
        ),
        member_table_ids=(1, 0, 1),
    )

    output = EnsembleTable.concatenate_columns((left, right))

    assert output.member(0).numerical.tolist() == [
        [3.0, 30.0],
        [4.0, 40.0],
    ]
    assert output.member(1).numerical.tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
    ]
    assert output.member(2).equal(output.member(0))


def test_concatenate_columns_regroups_different_layouts() -> None:
    left = EnsembleTable.from_tables(
        tables=(
            TableTensor.from_tensor(
                torch.tensor([[1.0], [2.0]]),
                columns=("left",),
            ),
            TableTensor.from_tensor(
                torch.tensor([[3.0], [4.0]]),
                columns=("left",),
            ),
        ),
        member_table_ids=(1, 0, 1),
    )
    right = EnsembleTable.from_table(
        TableTensor.from_tensor(
            torch.tensor([[10.0], [20.0]]),
            columns=("right",),
        ),
        num_members=3,
    )

    output = EnsembleTable.concatenate_columns((left, right))

    assert output.member(0).numerical.tolist() == [
        [3.0, 10.0],
        [4.0, 20.0],
    ]
    assert output.member(1).numerical.tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
    ]
    assert output.member(2).equal(output.member(0))


def test_concatenate_columns_rejects_different_member_counts() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))

    with pytest.raises(ValueError, match="different member counts"):
        EnsembleTable.concatenate_columns(
            (
                EnsembleTable.from_table(table, num_members=2),
                EnsembleTable.from_table(table, num_members=3),
            )
        )
