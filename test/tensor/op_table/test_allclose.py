import torch

from sdm import TableTensor

aten = torch.ops.aten


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48.0).view(2, 3, 4, 2),
    )


def test_allclose_overload() -> None:
    inp = make_table()
    close = TableTensor(
        columns={stype.value: names for stype, names in inp.columns.items()},
        numerical=inp.numerical + 1e-7,
    )

    assert aten.allclose.default(inp, close, 1e-5, 1e-6)
