import torch

from sdm import Stype, TableTensor
from sdm.processing import SelectColumns
from sdm.tensor import EnsembleTable


def test_select_first_columns() -> None:
    table = TableTensor(
        columns={
            "numerical": ("x0", "x1", "x2"),
            "datetime": ("d0", "d1", "d2"),
        },
        numerical=torch.arange(6, dtype=torch.float).view(2, 3),
        datetime=torch.arange(6, dtype=torch.int64).view(2, 3),
    )

    out = SelectColumns(max_columns=2, method="first").transform(table)
    assert out.equal(table.select_columns(("x0", "x1", "d0", "d1")))


def test_select_columns_round_robin_routes_members() -> None:
    table = TableTensor.from_tensor(
        torch.arange(10, dtype=torch.float).view(2, 5),
        columns=("x0", "x1", "x2", "x3", "x4"),
    )

    out = SelectColumns(
        max_columns=2,
        method="round_robin",
    ).fit_transform_ensemble(EnsembleTable(table, num_members=4))

    expected_columns = (
        ("x0", "x1"),
        ("x2", "x3"),
        ("x4", "x0"),
        ("x1", "x2"),
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
