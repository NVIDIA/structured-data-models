import torch

from sdm import EnsembleTable, TableTensor


def test_shared_representation() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    table = EnsembleTable(data, num_members=3)

    assert table.num_members == 3
    assert repr(table) == (
        "EnsembleTable(num_members=3, num_representations=1)"
    )
    assert len(tuple(table.iter_packed_representations())) == 1
    assert next(table.iter_packed_representations()).size() == (1, 2, 1)
    for member_id in range(3):
        assert table.representation(member_id).equal(data)


def test_from_representations_stacks_compatible_schemas() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    table = EnsembleTable.from_representations(
        representations=(first, second),
        member_representation_ids=(0, 1, 0, 1),
    )

    packed = tuple(table.iter_packed_representations())
    assert len(packed) == 1
    assert packed[0].size() == (2, 2, 1)
    assert table.representation(0).equal(first)
    assert table.representation(1).equal(second)
    assert table.representation(2).equal(first)
    assert table.representation(3).equal(second)


def test_from_representations_separates_incompatible_schemas() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]), columns=("first",)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]), columns=("second",)
    )

    table = EnsembleTable.from_representations(
        representations=(first, second),
        member_representation_ids=(0, 1),
    )

    assert len(tuple(table.iter_packed_representations())) == 2
    assert table.representation(0).columns == first.columns
    assert table.representation(1).columns == second.columns
