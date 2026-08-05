import pytest
import torch

from sdm.tensor import EnsembleTable, TableTensor


def test_shared_member_table() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_table = EnsembleTable(data, num_members=3)

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
        assert ensemble_table.table(member_id) is data


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
    ensemble_table = EnsembleTable(data, num_members=2)

    with pytest.raises(ValueError, match="one replacement per group"):
        ensemble_table.replace_groups(tuple(ensemble_table) * 2)


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
