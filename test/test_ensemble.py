import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.testing import withCUDA


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
        assert ensemble_table.table(member_id).equal(data)


def test_from_tables_stacks_compatible_schemas() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    groups = tuple(ensemble_table)
    assert len(groups) == 1
    assert groups[0].size() == (2, 2, 1)
    assert ensemble_table.table(0).equal(first)
    assert ensemble_table.table(1).equal(second)
    assert ensemble_table.table(2).equal(first)
    assert ensemble_table.table(3).equal(second)


def test_from_tables_separates_incompatible_schemas() -> None:
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
        member_table_ids=(0, 1),
    )

    assert len(tuple(ensemble_table)) == 2
    assert ensemble_table.table(0).columns == first.columns
    assert ensemble_table.table(1).columns == second.columns


def test_from_tables_separates_incompatible_block_sizes() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0], [5.0]]))

    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    assert len(tuple(ensemble_table)) == 2
    assert ensemble_table.table(0).size() == (2, 1)
    assert ensemble_table.table(1).size() == (3, 1)


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
    assert ensemble_table.table(0).equal(first)


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
    assert replaced.table(0).numerical.tolist() == [[-3.0], [-4.0]]
    assert replaced.table(1).numerical.tolist() == [[-1.0], [-2.0]]
    assert replaced.table(2).equal(replaced.table(0))
    assert ensemble_table.table(0).equal(second)


def test_replace_groups_rejects_group_count_mismatch() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_table = EnsembleTable.from_table(data, num_members=2)

    with pytest.raises(ValueError, match="one replacement per group"):
        ensemble_table.replace_groups(tuple(ensemble_table) * 2)


def test_gather_members_preserves_member_order() -> None:
    tables = tuple(
        TableTensor.from_tensor(torch.tensor([[value]], dtype=torch.float32))
        for value in range(4)
    )
    first = EnsembleTable.from_tables(tables[:2], member_table_ids=(0, 1))
    second = EnsembleTable.from_tables(tables[2:], member_table_ids=(0, 1))

    output = EnsembleTable.gather_members(
        tables=(first, second, first),
        member_ids=(1, 0, 0),
    )

    assert output.table(0).equal(tables[1])
    assert output.table(1).equal(tables[2])
    assert output.table(2).equal(tables[0])


def test_gather_members_rejects_source_count_mismatch() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))

    with pytest.raises(ValueError, match="one source member"):
        EnsembleTable.gather_members(
            tables=(EnsembleTable.from_table(table, num_members=2),),
            member_ids=(0, 1),
        )


def test_select_members_preserves_groups_and_order() -> None:
    tables = tuple(
        TableTensor.from_tensor(torch.tensor([[value]], dtype=torch.float32))
        for value in range(3)
    )
    ensemble_table = EnsembleTable.from_tables(
        tables=tables,
        member_table_ids=(2, 0, 1, 2),
    )

    output = ensemble_table.select_members((1, 3, 0))

    assert output.num_members == 3
    assert output.num_groups == 1
    assert next(iter(output)).size(0) == 2
    assert output.table(0).equal(tables[0])
    assert output.table(1).equal(tables[2])
    assert output.table(2).equal(tables[2])


@withCUDA
@pytest.mark.parametrize(
    ("member_ids", "shares_storage"),
    [
        ((1, 2, 3), True),
        ((0, 2, 4), True),
        ((1, 1, 3, 5, 3), True),
        ((4, 2, 0), False),
        ((0, 3, 4), False),
    ],
)
def test_select_members_slices_preserve_assignment(
    device: torch.device,
    member_ids: tuple[int, ...],
    shares_storage: bool,
) -> None:
    numerical = torch.arange(96, dtype=torch.float32, device=device).view(
        6, 8, 2
    )
    group = TableTensor.from_tensor(numerical[:, ::2])
    ensemble = EnsembleTable(
        groups=(group,),
        locations=tuple((0, position) for position in range(6)),
    )

    out = ensemble.select_members(member_ids)

    assert out.num_groups == 1
    assert next(iter(out)).size(0) == len(set(member_ids))
    assert out.num_members == len(member_ids)
    for member, source in enumerate(member_ids):
        assert out.table(member).equal(ensemble.table(source))
    selected = next(iter(out)).numerical
    assert (
        selected.untyped_storage().data_ptr()
        == numerical.untyped_storage().data_ptr()
    ) == shares_storage
    if shares_storage:
        out.table(0).numerical.fill_(-1)
        assert ensemble.table(member_ids[0]).numerical.eq(-1).all()


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

    assert output.table(0).numerical.tolist() == [
        [3.0, 30.0],
        [4.0, 40.0],
    ]
    assert output.table(1).numerical.tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
    ]
    assert output.table(2).equal(output.table(0))


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

    assert output.table(0).numerical.tolist() == [
        [3.0, 10.0],
        [4.0, 20.0],
    ]
    assert output.table(1).numerical.tolist() == [
        [1.0, 10.0],
        [2.0, 20.0],
    ]
    assert output.table(2).equal(output.table(0))


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
def test_concatenate_numerical_groups_preserves_layout_and_dtype(
    device: torch.device,
    num_rows: int,
    left_dtype: torch.dtype,
    right_dtype: torch.dtype,
) -> None:
    values = torch.arange(64, dtype=left_dtype, device=device).view(2, 8, 4)
    groups = tuple(
        TableTensor.from_tensor(
            (values + offset)[:, : num_rows * 2 : 2, ::2],
            columns=("left_0", "left_1"),
        )
        for offset in (0, 100)
    )
    left = EnsembleTable(
        groups=groups,
        locations=((1, 1), (0, 1), (1, 1), (0, 0), (1, 0)),
    )
    right_table = TableTensor.from_tensor(
        torch.arange(32, dtype=right_dtype, device=device).view(8, 4)[
            : num_rows * 2 : 2, ::2
        ],
        columns=("right_0", "right_1"),
    )
    right = EnsembleTable.from_table(right_table, num_members=5)
    originals = [group.numerical.clone() for group in (*groups, right_table)]

    out = EnsembleTable.concatenate_columns((left, right))

    assert out.num_members == 5
    assert out.num_groups == 1
    assert next(iter(out)).size() == (4, num_rows, 4)
    assert next(iter(out)).numerical.is_contiguous()
    for member_id in range(out.num_members):
        expected = torch.cat(
            (
                left.table(member_id).numerical,
                right.table(member_id).numerical,
            ),
            dim=-1,
        )
        actual = out.table(member_id)
        assert actual.columns[Stype.numerical] == (
            "left_0",
            "left_1",
            "right_0",
            "right_1",
        )
        torch.testing.assert_close(actual.numerical, expected)
    assert (
        out.table(0).numerical.data_ptr() == out.table(2).numerical.data_ptr()
    )
    for original, table in zip(originals, (*groups, right_table), strict=True):
        torch.testing.assert_close(table.numerical, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_concatenate_numerical_preserves_autocast(dtype: torch.dtype) -> None:
    left = EnsembleTable.from_tables(
        tables=tuple(
            TableTensor.from_tensor(
                torch.full((2, 1), value, dtype=dtype), columns=("left",)
            )
            for value in (1, 2)
        ),
        member_table_ids=(1, 0),
    )
    right = EnsembleTable.from_table(
        TableTensor.from_tensor(
            torch.ones(2, 1, dtype=dtype), columns=("right",)
        ),
        num_members=2,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if dtype == torch.float16:
            with pytest.raises(RuntimeError):
                EnsembleTable.concatenate_columns((left, right))
        else:
            out = EnsembleTable.concatenate_columns((left, right))
            assert out.table(0).numerical.dtype == torch.bfloat16
            torch.testing.assert_close(
                out.table(0).numerical,
                torch.tensor([[2, 1], [2, 1]], dtype=torch.bfloat16),
            )


def test_concatenate_numerical_preserves_heterogeneous_groups() -> None:
    left = EnsembleTable.from_tables(
        tables=tuple(
            TableTensor.from_tensor(
                torch.ones(2, 1, dtype=dtype), columns=("left",)
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
    assert out.table(0).numerical.dtype == torch.float32
    assert out.table(1).numerical.dtype == torch.float64
    assert (
        out.table(2).numerical.data_ptr() == out.table(0).numerical.data_ptr()
    )
    for member_id in range(out.num_members):
        torch.testing.assert_close(
            out.table(member_id).numerical,
            torch.ones(2, 2, dtype=left.table(member_id).numerical.dtype),
        )


def test_from_tables_separates_incompatible_layouts() -> None:
    dense = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    with torch.sparse.check_sparse_tensor_invariants():
        sparse = TableTensor.from_tensor(
            tensor=torch.tensor([[3.0], [4.0]]).to_sparse()
        )
        ensemble_table = EnsembleTable.from_tables(
            tables=(dense, sparse),
            member_table_ids=(0, 1),
        )

    assert len(tuple(ensemble_table)) == 2
