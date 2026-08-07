import torch

from sdm import TableTensor

aten = torch.ops.aten


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48.0).view(2, 3, 4, 2),
    )


def test_equal_overload() -> None:
    inp = make_table()
    same = inp.clone()
    different = TableTensor(
        columns={stype.value: names for stype, names in inp.columns.items()},
        numerical=inp.numerical + 1e-7,
    )

    assert aten.equal.default(inp, same)
    assert not aten.equal.default(inp, different)
