import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import ConstantFilter, Sequential, StandardScale


def _table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("constant", "variable", "all_nan"),
            "categorical": ("kind", "segment"),
        },
        numerical=torch.tensor(
            [
                [1.0, 2.0, float("nan")],
                [1.0, 5.0, float("nan")],
                [1.0, 8.0, float("nan")],
            ]
        ),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 0], [0, 1], [0, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["small", "large"]),
            ),
        ),
    )


def test_constant_filter_removes_constant_columns_and_metadata() -> None:
    table = _table()

    output = ConstantFilter().fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("variable",)
    assert output.columns[Stype.categorical] == ("segment",)
    assert torch.equal(output.numerical, table.numerical[:, [1]])
    assert torch.equal(
        output.categorical.as_tensor(),
        table.categorical.as_tensor()[:, [1]],
    )


def test_constant_filter_keeps_all_columns_for_few_samples() -> None:
    table = _table()

    output = ConstantFilter(threshold=3).fit_transform(table)

    assert output is table


def test_constant_filter_noop_returns_input_table() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    output = ConstantFilter().fit_transform(table)

    assert output is table


def test_constant_filter_composes_before_block_processor() -> None:
    table = _table()

    output = Sequential(ConstantFilter(), StandardScale()).fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("variable",)
    assert torch.allclose(output.numerical.mean(dim=0), torch.zeros(1))


def test_constant_filter_rejects_non_table_input() -> None:
    processor = ConstantFilter()

    with pytest.raises(TypeError, match="Expected ConstantFilter input"):
        processor.fit(torch.ones(2, 3))
