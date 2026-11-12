# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import cast

import pytest
import torch

from sdm import (
    CategoricalTensor,
    EnsembleData,
    EnsembleTable,
    EnsembleTensor,
    TableTensor,
)


@pytest.mark.parametrize(
    "values", [("first", "second"), (["first"], ["second"])]
)
def test_data_member_selection(
    values: tuple[str, str] | tuple[list[str], list[str]],
) -> None:
    ensemble = EnsembleData(
        groups=values,
        locations=((1, 0), (0, 0), (1, 0)),
    )

    assert tuple(ensemble) == (values[1], values[0], values[1])
    assert ensemble[0] is ensemble[2]
    selected = ensemble[2, 0, 2]
    assert tuple(selected) == (values[1],) * 3
    assert selected[0] is values[1]
    assert tuple(ensemble[1:]) == (values[0], values[1])
    assert ensemble[-1] is values[1]
    assert len(ensemble[()]) == 0


def test_column_names() -> None:
    batch = TableTensor.from_tensor(
        tensor=torch.zeros(2, 3, 2),
        columns=("first", "second"),
    )
    other = TableTensor.from_tensor(
        tensor=torch.ones(1, 4, 1),
        columns=("other",),
    )
    ensemble = EnsembleTable(
        groups=(batch, other),
        locations=((0, 1), (1, 0), (0, 0), (0, 1)),
    )

    names = ensemble.column_names

    assert isinstance(names, EnsembleData)
    assert tuple(names) == (
        frozenset(("first", "second")),
        frozenset(("other",)),
        frozenset(("first", "second")),
        frozenset(("first", "second")),
    )
    assert names[0] is names[2] is names[3]
    assert len(EnsembleTable(groups=(), locations=()).column_names) == 0


@pytest.mark.parametrize("columns", ["keep", ("keep",), iter(("keep",)), ()])
def test_select_columns(columns: str | Iterable[str]) -> None:
    batch = TableTensor.from_tensor(
        tensor=torch.arange(12, dtype=torch.float32).view(2, 3, 2),
        columns=("keep", "drop"),
    )
    other = TableTensor.from_tensor(
        tensor=torch.ones(1, 4, 3),
        columns=("drop", "keep", "extra"),
    )
    ensemble = EnsembleTable(
        groups=(batch, other),
        locations=((0, 1), (1, 0), (0, 0), (0, 1)),
    )
    expected = () if columns == () else ("keep",)

    selected = ensemble.select_columns(columns)

    assert len(selected) == len(ensemble)
    assert selected.num_groups == ensemble.num_groups
    for group, source in zip(
        selected._iter_groups(), (batch, other), strict=True
    ):
        assert group.equal(source.select_columns(expected))
        assert group.size(0) == source.size(0)
    for member_id in range(len(ensemble)):
        assert selected[member_id].equal(
            ensemble[member_id].select_columns(expected)
        )


def test_data_does_not_infer_tensor_batching() -> None:
    tensor = torch.arange(6).view(2, 3)
    values = EnsembleData(groups=(tensor,), locations=((0, 0), (0, 0)))
    members = EnsembleTensor(groups=(tensor,), locations=((0, 0), (0, 1)))

    assert values[0] is tensor
    assert values[1] is tensor
    assert members[0].equal(tensor[0])
    assert members[1].equal(tensor[1])


def test_tensor_member_selection() -> None:
    first = torch.tensor([[1.0], [2.0], [3.0]])
    second = torch.tensor([[4.0]])
    ensemble = EnsembleTensor(
        groups=(first, second),
        locations=((0, 0), (1, 0), (0, 2), (0, 1), (1, 0)),
    )

    selected = ensemble[4, 1, 2, 4, 0]

    assert isinstance(selected, EnsembleTensor)
    assert len(selected) == 5
    torch.testing.assert_close(
        torch.stack([selected[index] for index in range(len(selected))]),
        torch.tensor([[4.0], [4.0], [3.0], [4.0], [1.0]]),
    )
    assert selected[0].data_ptr() == selected[1].data_ptr()
    assert selected[0].data_ptr() == selected[3].data_ptr()
    sliced = ensemble[:3:2]
    torch.testing.assert_close(torch.stack((sliced[0], sliced[1])), first[::2])
    assert sliced[0].data_ptr() == first[0].data_ptr()
    assert ensemble[:] is ensemble
    assert len(ensemble[()]) == 0


def test_shared_member_table() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_table = EnsembleTable.from_table(data, num_members=3)

    assert len(ensemble_table) == 3
    assert ensemble_table.num_groups == 1
    assert repr(ensemble_table) == (
        "EnsembleTable(num_members=3, num_groups=1)"
    )
    groups = tuple(ensemble_table._iter_groups())
    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert next(ensemble_table._iter_groups()) is groups[0]
    for member_id in range(3):
        assert ensemble_table[member_id].equal(data)


def test_from_tables_keeps_tables_separate() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    groups = tuple(ensemble_table._iter_groups())
    assert len(groups) == 2
    assert all(group.size() == (1, 2, 1) for group in groups)
    assert ensemble_table[0].equal(first)
    assert ensemble_table[1].equal(second)
    assert ensemble_table[2].equal(first)
    assert ensemble_table[3].equal(second)


def test_from_tables_ignores_unused_tables() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0,),
    )

    groups = tuple(ensemble_table._iter_groups())

    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert ensemble_table[0].equal(first)


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
            for group in ensemble_table._iter_groups()
        ]
    )

    assert len(replaced) == 3
    assert replaced.num_groups == ensemble_table.num_groups
    assert replaced[0].numerical.tolist() == [[-3.0], [-4.0]]
    assert replaced[1].numerical.tolist() == [[-1.0], [-2.0]]
    assert replaced[2].equal(replaced[0])
    assert ensemble_table[0].equal(second)


def test_replace_groups_rejects_group_count_mismatch() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_table = EnsembleTable.from_table(data, num_members=2)

    with pytest.raises(ValueError, match="one replacement per group"):
        ensemble_table.replace_groups(tuple(ensemble_table._iter_groups()) * 2)


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
    assert all(group.size(0) == 2 for group in output._iter_groups())
    for member_id, table in enumerate(tables):
        assert output[member_id].equal(table)


@pytest.mark.parametrize("shared_categories", [False, True])
def test_replace_tables_preserves_category_identity(
    shared_categories: bool,
) -> None:
    categories = torch.tensor([10, 20])
    second_categories = categories if shared_categories else categories.clone()
    tables = tuple(
        TableTensor(
            categorical=CategoricalTensor(
                code=code,
                categories=(vocabulary,),
            )
        )
        for code, vocabulary in (
            (torch.tensor([[0], [-1]]), categories),
            (torch.tensor([[1], [0]]), second_categories),
        )
    )
    ensemble_table = EnsembleTable.from_table(tables[0], num_members=3)
    output = ensemble_table.replace_tables(
        tables=tables,
        member_table_ids=(0, 1, 0),
    )

    assert output.num_groups == (1 if shared_categories else 2)
    for member_id, table_id in enumerate((0, 1, 0)):
        actual = output[member_id].categorical
        expected = tables[table_id].categorical
        assert actual.categories[0] is expected.categories[0]
        torch.testing.assert_close(actual.code, expected.code)


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

    assert output[0].equal(tables[1])
    assert output[1].equal(tables[2])
    assert output[2].equal(tables[1])
    assert next(output._iter_groups()).size(0) == 2


def test_gather_members_rejects_source_count_mismatch() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable.from_table(table, num_members=2)

    with pytest.raises(ValueError, match="one source member"):
        ensemble_table.gather_members(
            tables=(ensemble_table,),
            member_ids=(0, 1),
        )


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

    assert output[0].numerical.tolist() == [
        [3.0, 30.0],
        [4.0, 40.0],
    ]
    assert output[1].numerical.tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
    ]
    assert output[2].equal(output[0])


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

    assert output[0].numerical.tolist() == [
        [3.0, 10.0],
        [4.0, 20.0],
    ]
    assert output[1].numerical.tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
    ]
    assert output[2].equal(output[0])


def test_concatenate_columns_rejects_different_member_counts() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))

    with pytest.raises(ValueError, match="different member counts"):
        EnsembleTable.concatenate_columns(
            (
                EnsembleTable.from_table(table, num_members=2),
                EnsembleTable.from_table(table, num_members=3),
            )
        )
