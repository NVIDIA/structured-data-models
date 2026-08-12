import torch

from sdm import TableTensor
from sdm.processing import SelectColumns


def test_select_first_columns() -> None:
    table = TableTensor(
        numerical=torch.arange(6, dtype=torch.float).view(2, 3),
        datetime=torch.arange(6, dtype=torch.int64).view(2, 3),
    )

    out = SelectColumns(max_columns=2, method="first").transform(table)
    assert out.equal(table.select_columns(("num_0", "num_1", "dt_0", "dt_1")))
