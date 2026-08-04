import torch

from sdm import EnsembleTable, TableTensor


def test_shared_member_table() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    table = EnsembleTable(data, num_members=3)

    assert table.num_members == 3
    assert repr(table) == "EnsembleTable(num_members=3, num_member_tables=1)"
    assert len(tuple(table.member_groups())) == 1
    assert next(table.member_groups()).size() == (1, 2, 1)
    for member_id in range(3):
        assert table.member_table(member_id).equal(data)


def test_from_member_tables_stacks_compatible_schemas() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    table = EnsembleTable.from_member_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    groups = tuple(table.member_groups())
    assert len(groups) == 1
    assert groups[0].size() == (2, 2, 1)
    assert table.member_table(0).equal(first)
    assert table.member_table(1).equal(second)
    assert table.member_table(2).equal(first)
    assert table.member_table(3).equal(second)


def test_from_member_tables_separates_incompatible_schemas() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]), columns=("first",)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]), columns=("second",)
    )

    table = EnsembleTable.from_member_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    assert len(tuple(table.member_groups())) == 2
    assert table.member_table(0).columns == first.columns
    assert table.member_table(1).columns == second.columns
