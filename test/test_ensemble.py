# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.testing import withCUDA


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


def test_getitem_preserves_groups_and_order() -> None:
    tables = tuple(
        TableTensor.from_tensor(torch.tensor([[value]], dtype=torch.float32))
        for value in range(3)
    )
    ensemble_table = EnsembleTable(
        groups=(cast(TableTensor, torch.stack(tables)),),
        locations=((0, 2), (0, 0), (0, 1), (0, 2)),
    )

    output = ensemble_table[1, 3, 0]

    assert len(output) == 3
    assert output.num_groups == 1
    assert next(output._iter_groups()).size(0) == 2
    assert output[0].equal(tables[0])
    assert output[1].equal(tables[2])
    assert output[2].equal(tables[2])

    output = ensemble_table[1:4:2]

    assert len(output) == 2
    assert output.num_groups == 1
    assert next(output._iter_groups()).size(0) == 2
    assert output[0].equal(tables[0])
    assert output[1].equal(tables[2])


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


@withCUDA
@pytest.mark.parametrize("num_rows", [0, 4])
@pytest.mark.parametrize(
    ("left_dtype", "right_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.float64),
        (torch.bfloat16, torch.float32),
    ],
)
def test_concatenate_numerical_groups_preserves_members(
    device: torch.device,
    num_rows: int,
    left_dtype: torch.dtype,
    right_dtype: torch.dtype,
) -> None:
    values = torch.arange(64, dtype=left_dtype, device=device).view(2, 8, 4)
    groups = tuple(
        TableTensor.from_tensor(
            tensor=(values + offset)[:, : num_rows * 2 : 2, ::2],
            columns=("left_0", "left_1"),
        )
        for offset in (0, 100)
    )
    left = EnsembleTable(
        groups=groups,
        locations=((1, 1), (0, 1), (1, 1), (0, 0), (1, 0)),
    )
    right_table = TableTensor.from_tensor(
        tensor=torch.arange(32, dtype=right_dtype, device=device).view(8, 4)[
            : num_rows * 2 : 2, ::2
        ],
        columns=("right_0", "right_1"),
    )
    right = EnsembleTable.from_table(right_table, num_members=5)
    originals = [group.numerical.clone() for group in (*groups, right_table)]

    out = EnsembleTable.concatenate_columns((left, right))

    assert len(out) == 5
    assert out.num_groups == 4
    for member_id in range(len(out)):
        expected = torch.cat(
            (left[member_id].numerical, right[member_id].numerical), dim=-1
        )
        assert out[member_id].columns[Stype.numerical] == (
            "left_0",
            "left_1",
            "right_0",
            "right_1",
        )
        torch.testing.assert_close(out[member_id].numerical, expected)

    # Shared members alias; independent members and inputs remain unchanged.
    independent = out[1].numerical.clone()
    out[0].numerical.fill_(-1)
    torch.testing.assert_close(out[2].numerical, out[0].numerical)
    torch.testing.assert_close(out[1].numerical, independent)
    for original, table in zip(originals, (*groups, right_table), strict=True):
        torch.testing.assert_close(table.numerical, original)


def test_concatenate_numerical_preserves_heterogeneous_groups() -> None:
    left = EnsembleTable.from_tables(
        tables=tuple(
            TableTensor.from_tensor(
                tensor=torch.ones(2, 1, dtype=dtype), columns=("left",)
            )
            for dtype in (torch.float32, torch.float64)
        ),
        member_table_ids=(0, 1, 0),
    )
    right = EnsembleTable.from_table(
        TableTensor.from_tensor(torch.ones(2, 1), columns=("right",)),
        num_members=3,
    )

    out = EnsembleTable.concatenate_columns((left, right))

    assert out.num_groups == 2
    for member_id in range(len(out)):
        torch.testing.assert_close(
            out[member_id].numerical,
            torch.ones(2, 2, dtype=left[member_id].numerical.dtype),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_concatenate_numerical_preserves_autocast(dtype: torch.dtype) -> None:
    left = EnsembleTable.from_tables(
        tables=tuple(
            TableTensor.from_tensor(
                tensor=torch.full((2, 1), value, dtype=dtype),
                columns=("left",),
            )
            for value in (1, 2)
        ),
        member_table_ids=(1, 0),
    )
    right = EnsembleTable.from_table(
        TableTensor.from_tensor(torch.ones(2, 1), columns=("right",)),
        num_members=2,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = EnsembleTable.concatenate_columns((left, right))
        for member_id in range(len(out)):
            expected = torch.cat(
                (left[member_id].numerical, right[member_id].numerical), dim=-1
            )
            torch.testing.assert_close(out[member_id].numerical, expected)


def test_concatenate_numerical_preserves_grad() -> None:
    values = torch.randn(2, 3, 1, requires_grad=True)
    left = EnsembleTable.from_tables(
        tables=tuple(
            TableTensor(numerical=member, columns={"numerical": ("left",)})
            for member in values
        ),
        member_table_ids=(1, 0, 1),
    )
    right = EnsembleTable.from_table(
        TableTensor.from_tensor(torch.ones(3, 1), columns=("right",)),
        num_members=3,
    )
    out = EnsembleTable.concatenate_columns((left, right))
    torch.stack([member.numerical.sum() for member in out]).sum().backward()
    torch.testing.assert_close(
        values.grad, values.new_tensor([1, 2]).view(2, 1, 1).expand_as(values)
    )
