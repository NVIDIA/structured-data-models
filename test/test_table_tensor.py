import torch
from schemafm import Stype, TableTensor


def test_init() -> None:
    table = TableTensor(
        data=torch.randn(2, 2),
        names=("age", "fraud"),
        stypes=("numerical", "categorical"),
        colptr=range(3),
    )

    assert table.size() == (2, 2)
    assert table.column_names == ("age", "fraud")
    assert table.stypes == (Stype.numerical, Stype.categorical)
    assert table.colptr.equal(torch.tensor([0, 1, 2]))
