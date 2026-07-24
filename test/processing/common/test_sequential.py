import pickle
from textwrap import dedent
from typing import Any, cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    Sequential,
    ShuffleColumns,
    Softmax,
    Standardize,
)


def _mixed_table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    n_rows = numerical.shape[0]
    categorical = CategoricalTensor(
        data=(torch.arange(n_rows) % 2).unsqueeze(1),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    return TableTensor(
        columns={
            "numerical": ("x0", "x1"),
            "categorical": ("kind",),
        },
        numerical=numerical,
        categorical=categorical,
    )


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return TableTensor.from_tensor(numerical, columns=("x0", "x1"))


def _add_one(table: TableTensor) -> TableTensor:
    return table.replace_blocks(numerical=table.numerical + 1)


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Sequential().transform(table) is table
    assert Sequential().fit_transform(table) is table
    assert Sequential().inverse_transform(table) is table


def test_pipeline_transforms_numerical() -> None:
    table = _table()

    output = Sequential(Standardize()).fit_transform(table)

    assert not torch.equal(output.numerical, table.numerical)


def test_pipeline_accepts_lambda() -> None:
    table = _table()
    pipeline = Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical.square())
    )

    output = pipeline.transform(table)

    assert not pipeline.requires_fit
    assert torch.equal(output.numerical, table.numerical.square())


def test_pipeline_accepts_regular_callable() -> None:
    table = _table()
    pipeline = Sequential(_add_one)

    assert pipeline.fit(table) is pipeline
    restored = pickle.loads(pickle.dumps(pipeline))

    assert torch.equal(
        restored.transform(table).numerical,
        table.numerical + 1,
    )


def test_pipeline_mixes_processors_and_callables() -> None:
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [4.0, 8.0]],
        )
    )
    pipeline = Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical.square()),
        Standardize(),
    )
    expected = Standardize().fit_transform(
        table.replace_blocks(numerical=table.numerical.square())
    )

    assert pipeline.fit(table) is pipeline
    output = pipeline.transform(table)

    torch.testing.assert_close(output.numerical, expected.numerical)


def test_pipeline_accepts_nested_sequential_with_callable() -> None:
    table = _table()
    pipeline = Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical + 1),
        Sequential(
            lambda table: table.replace_blocks(numerical=table.numerical * 2)
        ),
    )

    output = pipeline.transform(table)

    assert torch.equal(output.numerical, (table.numerical + 1) * 2)


def test_pipeline_rejects_invalid_step() -> None:
    with pytest.raises(
        TypeError,
        match=r"step 0.*Processor or callable.*object",
    ):
        Sequential(cast(Any, object()))


def test_pipeline_passes_generator_to_steps() -> None:
    table = _table(torch.arange(200.0).view(100, 2))

    def _fit(seed: int) -> tuple[ShuffleColumns, QuantileTransform]:
        permute = ShuffleColumns(method="random")
        quantile = QuantileTransform(n_quantiles=6, subsample=32)
        Sequential(permute, quantile).fit(
            table,
            generator=torch.Generator().manual_seed(seed),
        )
        return permute, quantile

    first_permute, first_quantile = _fit(0)
    second_permute, second_quantile = _fit(0)

    assert torch.equal(first_permute.permutation, second_permute.permutation)
    assert torch.equal(first_quantile.quantiles, second_quantile.quantiles)


def test_repr() -> None:
    assert repr(Sequential()) == "Sequential()"
    assert repr(Sequential(Standardize(), PowerTransform())) == dedent("""\
        Sequential(
          Standardize(),
          PowerTransform(),
        )""")
    assert repr(Sequential(lambda table: table)) == dedent("""\
        Sequential(
          lambda,
        )""")


def test_pipeline_checks_fitted_state() -> None:
    pipeline = Sequential(Softmax(), Standardize())

    with pytest.raises(RuntimeError, match="'Sequential' is not fitted"):
        pipeline.transform(_table())


def test_pipeline_rejects_unsupported_stype() -> None:
    pipeline = Sequential(Standardize())

    with pytest.raises(ValueError, match="categorical"):
        pipeline.fit_transform(_mixed_table())


def test_inverse_transform_rejects_non_invertible_step() -> None:
    processor = Sequential(ImputeMean())
    transformed = processor.fit_transform(_table())

    with pytest.raises(
        AttributeError,
        match=r"ImputeMean.*inverse_transform",
    ):
        processor.inverse_transform(transformed)


def test_inverse_transform_runs_steps_in_reverse_order() -> None:
    # PowerTransform and Standardize do not commute, so the round trip only
    # reconstructs the input if the inverse applies the steps in reverse.
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
    )

    pipeline = Sequential(PowerTransform(), Standardize())
    transformed = pipeline.fit_transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert torch.allclose(restored.numerical, table.numerical, atol=1e-4)
