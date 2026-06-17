import torch
from schemafm import Stype, TableTensor


def test_init() -> None:
    table = TableTensor(
        data=torch.randn(2, 2),
        col_names=("age", "fraud"),
        stypes=("numerical", "categorical"),
        colptr=range(3),
    )

    assert table.size() == (2, 2)
    assert table.col_names == ("age", "fraud")
    assert table.stypes == (Stype.numerical, Stype.categorical)
    assert table.colptr.equal(torch.tensor([0, 1, 2]))


def test_shape_preserving_ops() -> None:
    def assert_metadata(out: TableTensor) -> None:
        assert isinstance(out, TableTensor)
        assert out.col_names == table.col_names
        assert out.stypes == table.stypes
        assert out.colptr.equal(table.colptr)

    table = TableTensor(
        data=torch.randn(2, 2),
        col_names=("age", "fraud"),
        stypes=("numerical", "categorical"),
        colptr=range(3),
    )

    for out in (
        table.clone(),
        table.detach(),
        table.to(torch.float64),
    ):
        assert_metadata(out)

    assert not table.is_shared()
    assert not table.as_tensor().is_shared()
    assert table.share_memory_() is table
    assert table.is_shared()
    assert table.as_tensor().is_shared()

    if torch.cuda.is_available():
        assert not out.is_pinned()
        assert not out.as_tensor().is_pinned()
        out = table.pin_memory()
        assert_metadata(out)
        assert out.is_pinned()
        assert out.as_tensor().is_pinned()

    table = TableTensor(
        data=torch.randn(2, 2, requires_grad=True),
        col_names=("age", "fraud"),
        stypes=("numerical", "categorical"),
        colptr=range(3),
    )
    assert table.requires_grad
    assert table.as_tensor().requires_grad
    assert table.detach_() is table
    assert not table.requires_grad
    assert not table.as_tensor().requires_grad
