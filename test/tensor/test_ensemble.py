import torch

from sdm.tensor import EnsembleTensor, TableTensor


def test_shared_member_table() -> None:
    data = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    ensemble_tensor = EnsembleTensor(data, num_members=3)

    assert ensemble_tensor.num_members == 3
    assert repr(ensemble_tensor) == (
        "EnsembleTensor(num_members=3, num_groups=1)"
    )
    groups = tuple(ensemble_tensor)
    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert next(iter(ensemble_tensor)) is groups[0]
    for member_id in range(3):
        assert ensemble_tensor.table(member_id).equal(data)


def test_from_tables_stacks_compatible_schemas() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))

    ensemble_tensor = EnsembleTensor.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1, 0, 1),
    )

    groups = tuple(ensemble_tensor)
    assert len(groups) == 1
    assert groups[0].size() == (2, 2, 1)
    assert ensemble_tensor.table(0).equal(first)
    assert ensemble_tensor.table(1).equal(second)
    assert ensemble_tensor.table(2).equal(first)
    assert ensemble_tensor.table(3).equal(second)


def test_from_tables_separates_incompatible_schemas() -> None:
    first = TableTensor.from_tensor(
        tensor=torch.tensor([[1.0], [2.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        tensor=torch.tensor([[3.0], [4.0]]),
        columns=("second",),
    )

    ensemble_tensor = EnsembleTensor.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    assert len(tuple(ensemble_tensor)) == 2
    assert ensemble_tensor.table(0).columns == first.columns
    assert ensemble_tensor.table(1).columns == second.columns


def test_from_tables_separates_incompatible_block_sizes() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0], [5.0]]))

    ensemble_tensor = EnsembleTensor.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    assert len(tuple(ensemble_tensor)) == 2
    assert ensemble_tensor.table(0).size() == (2, 1)
    assert ensemble_tensor.table(1).size() == (3, 1)


def test_from_tables_ignores_unused_tables() -> None:
    first = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    second = TableTensor.from_tensor(torch.tensor([[3.0], [4.0]]))
    ensemble_tensor = EnsembleTensor.from_tables(
        tables=(first, second),
        member_table_ids=(0,),
    )

    groups = tuple(ensemble_tensor)

    assert len(groups) == 1
    assert groups[0].size() == (1, 2, 1)
    assert ensemble_tensor.table(0).equal(first)


def test_from_tables_separates_incompatible_layouts() -> None:
    dense = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    with torch.sparse.check_sparse_tensor_invariants():
        sparse = TableTensor.from_tensor(
            tensor=torch.tensor([[3.0], [4.0]]).to_sparse()
        )
        ensemble_tensor = EnsembleTensor.from_tables(
            tables=(dense, sparse),
            member_table_ids=(0, 1),
        )

    assert len(tuple(ensemble_tensor)) == 2
