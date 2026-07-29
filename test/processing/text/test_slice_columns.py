import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing.text.slice_columns import SliceColumns


def _table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("x0", "x1", "x2")},
        numerical=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )


def test_slice_features_keeps_leading_columns() -> None:
    output = SliceColumns(max_columns=2).transform(_table())

    assert output.columns[Stype.numerical] == ("x0", "x1")
    assert torch.equal(
        output.numerical, torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    )


def test_slice_features_passes_through_when_narrower() -> None:
    table = _table()

    output = SliceColumns(max_columns=99).transform(table)

    assert output.columns == table.columns
    assert torch.equal(output.numerical, table.numerical)


def test_slice_features_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        SliceColumns(max_columns=0)
