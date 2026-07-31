import torch

from sdm import Stype, TableTensor
from sdm.processing.common import SliceColumns


def _table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0", "x1", "x2"),
            "datetime": ("d0", "d1", "d2"),
        },
        numerical=torch.arange(6, dtype=torch.float).view(2, 3),
        datetime=torch.arange(6, dtype=torch.int64).view(2, 3),
    )


def test_slice_columns_keeps_leading_columns_per_stype() -> None:
    table = _table()

    output = SliceColumns(
        max_columns={
            Stype.numerical: 2,
            "datetime": 1,
        }
    ).transform(table)

    expected = table.select_columns(("x0", "x1", "d0"))
    assert output.equal(expected)


def test_slice_columns_applies_integer_to_every_stype() -> None:
    table = _table()

    output = SliceColumns(max_columns=1).transform(table)

    expected = table.select_columns(("x0", "d0"))
    assert output.equal(expected)


def test_slice_columns_passes_through_unconfigured_stypes() -> None:
    table = _table()

    output = SliceColumns(max_columns={"numerical": 1}).transform(table)

    expected = table.select_columns(("x0", "d0", "d1", "d2"))
    assert output.equal(expected)
