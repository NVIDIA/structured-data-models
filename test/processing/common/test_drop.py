import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor


def _mixed_table() -> TableTensor:
    return TableTensor(
        numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def test_drop_stypes_removes_configured_stypes() -> None:
    table = _mixed_table()

    output = sp.DropStypes(Stype.numerical).fit_transform(table)

    assert output.columns[Stype.numerical] == ()
    assert output.columns[Stype.categorical] == ("cat_0",)
    assert torch.equal(output.categorical.code, table.categorical.code)


def test_drop_stypes_noops_without_matching_stypes() -> None:
    table = _mixed_table()

    assert sp.DropStypes(Stype.id).fit_transform(table) is table
