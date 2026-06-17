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
    table = TableTensor(
        data=torch.randn(2, 2),
        col_names=("age", "fraud"),
        stypes=("numerical", "categorical"),
        colptr=range(3),
    )

    def assert_metadata(out: TableTensor) -> None:
        return
        assert isinstance(out, TableTensor)
        assert out.col_names == table.col_names
        assert out.stypes == table.stypes
        assert out.colptr.equal(table.colptr)

    for out in (
        table.clone(),
        table.detach(),
        table.to(torch.float64),
        table.view(table.size()),
    ):
        assert_metadata(out)

    # assert table.share_memory_() is table
    # assert table.is_shared()

    if torch.cuda.is_available():
        out = table.pin_memory()
        assert_metadata(out)
        # assert out.is_pinned()
