import torch

from sdm.tensor import EnsembleTable, TableTensor


def test_shared_member_table() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    table = EnsembleTable(data, num_members=3)

    assert table.num_members == 3
    assert repr(table) == "EnsembleTable(num_members=3, num_groups=1)"
    groups = tuple(table.groups())
    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert next(iter(table)) is groups[0]
    for member_id in range(3):
        assert table.table(member_id).equal(data)


def test_from_tables_stacks_compatible_schemas() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    groups = tuple(table.groups())
    assert len(groups) == 1
    assert groups[0].size() == (2, 2, 1)
    assert table.table(0).equal(first)
    assert table.table(1).equal(second)
    assert table.table(2).equal(first)
    assert table.table(3).equal(second)


def test_from_tables_separates_incompatible_schemas() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]), columns=("first",)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]), columns=("second",)
    )

    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    assert len(tuple(table.groups())) == 2
    assert table.table(0).columns == first.columns
    assert table.table(1).columns == second.columns


def test_from_tables_separates_incompatible_block_sizes() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0], [5.0]]))

    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    assert len(tuple(table.groups())) == 2
    assert table.table(0).size() == (2, 1)
    assert table.table(1).size() == (3, 1)
