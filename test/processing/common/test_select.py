import pytest
import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing import SelectColumns


def test_select_columns_rejects_negative_max_columns() -> None:
    with pytest.raises(ValueError, match="max_columns must be positive"):
        SelectColumns(max_columns=-1)


def test_select_first_columns() -> None:
    table = TableTensor(
        numerical=torch.arange(6, dtype=torch.float).view(2, 3),
        datetime=torch.arange(6, dtype=torch.int64).view(2, 3),
    )

    out = SelectColumns(max_columns=2, method="first").transform(table)
    assert out.equal(table.select_columns(("num_0", "num_1", "dt_0", "dt_1")))


def test_select_columns_round_robin_routes_members() -> None:
    table = TableTensor(
        numerical=torch.arange(10, dtype=torch.float).view(2, 5),
    )

    out = SelectColumns(
        max_columns=2,
        method="round_robin",
    ).fit_transform_ensemble(EnsembleTable.from_table(table, num_members=4))

    expected_columns = (
        ("num_0", "num_1"),
        ("num_2", "num_3"),
        ("num_4", "num_0"),
        ("num_1", "num_2"),
    )
    for member_id, columns in enumerate(expected_columns):
        indices = [
            table.columns[Stype.numerical].index(column) for column in columns
        ]
        expected = TableTensor.from_tensor(
            table.numerical[..., indices],
            columns=columns,
        )
        assert out.table(member_id).equal(expected)
