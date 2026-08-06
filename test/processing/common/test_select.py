import torch

from sdm import TableTensor
from sdm.processing import SelectColumns


def test_select_first_columns() -> None:
    table = TableTensor(
        columns={
            "numerical": ("x0", "x1", "x2"),
            "datetime": ("d0", "d1", "d2"),
        },
        numerical=torch.arange(6, dtype=torch.float).view(2, 3),
        datetime=torch.arange(6, dtype=torch.int64).view(2, 3),
    )

    out = SelectColumns(max_columns=2, mode="first").transform(table)
    assert out.equal(table.select_columns(("x0", "x1", "d0", "d1")))
