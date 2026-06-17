import torch
from schemafm import Stype, TableTensor


def test_init() -> None:
    table = TableTensor(
        data=torch.randn(2, 2),
        names=("age", "fraud"),
        stypes=("numerical", "categorical"),
    )

    assert table.size() == (2, 2)
    assert table.column_names == ("age", "fraud")
    assert table.stypes == (Stype.numerical, Stype.categorical)
    assert torch.equal(table.colptr, torch.tensor([0, 1, 2]))


def test_init_with_colptr() -> None:
    table = TableTensor(
        data=torch.randn(2, 5),
        names=("age", "embedding", "fraud"),
        stypes=("numerical", "numerical", "categorical"),
        colptr=(0, 1, 4, 5),
    )

    assert table.size() == (2, 5)
    assert table.column_names == ("age", "embedding", "fraud")
    assert table.stypes == (
        Stype.numerical,
        Stype.numerical,
        Stype.categorical,
    )
    assert torch.equal(table.colptr, torch.tensor([0, 1, 4, 5]))


def test_tensor_flatten_roundtrip_preserves_colptr() -> None:
    table = TableTensor(
        data=torch.randn(2, 5),
        names=("age", "embedding", "fraud"),
        stypes=("numerical", "numerical", "categorical"),
        colptr=(0, 1, 4, 5),
    )

    attrs, ctx = table.__tensor_flatten__()
    out = TableTensor.__tensor_unflatten__(
        inner_tensors={
            "_data": table.as_tensor(),
            "_colptr": table.colptr,
        },
        ctx=ctx,
        outer_size=tuple(table.size()),
        outer_stride=tuple(table.stride()),
    )

    assert attrs == ["_data", "_colptr"]
    assert out.column_names == table.column_names
    assert out.stypes == table.stypes
    assert torch.equal(out.colptr, table.colptr)
