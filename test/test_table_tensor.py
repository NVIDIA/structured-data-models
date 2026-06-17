import torch

from schemafm import Stype, TableTensor


def test_init():
    table = TableTensor(
        data=torch.randn(2, 2),
        names=("age", "fraud"),
        stypes=("numerical", "categorical"),
    )

    assert table.size() == (2, 2)
    assert table.column_names == ("age", "fraud")
    assert table.stypes == (Stype.numerical, Stype.categorical)
