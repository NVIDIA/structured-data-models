import pickle
from textwrap import dedent
from typing import Any, cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    FeaturePermute,
    MeanImpute,
    Power,
    Processor,
    Quantile,
    Sequential,
    SoftmaxTemperature,
    StandardScale,
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


def _multiply_by_two(table: TableTensor) -> TableTensor:
    return table.replace_blocks(numerical=table.numerical * 2)


class _AddOne(Processor):
    supported_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return _add_one(table)


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Sequential().transform(table) is table
    assert Sequential().fit_transform(table) is table
    assert Sequential().inverse_transform(table) is table


def test_pipeline_transforms_numerical() -> None:
    table = _table()

    output = Sequential(StandardScale()).fit_transform(table)

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

    restored = pickle.loads(pickle.dumps(pipeline))

    assert torch.equal(
        restored.fit_transform(table).numerical,
        table.numerical + 1,
    )


def test_pipeline_mixes_processors_and_callables() -> None:
    table = _table()
    pipeline = Sequential(StandardScale(), _add_one)

    output = pipeline.fit_transform(table)

    assert pipeline.requires_fit
    torch.testing.assert_close(
        output.numerical.mean(dim=0),
        torch.ones(2),
    )


def test_pipeline_accepts_nested_sequential_with_callable() -> None:
    table = _table()
    pipeline = Sequential(_add_one, Sequential(_multiply_by_two))

    output = pipeline.transform(table)

    assert torch.equal(output.numerical, (table.numerical + 1) * 2)


def test_pipeline_rejects_invalid_step() -> None:
    with pytest.raises(
        TypeError,
        match=r"step 0.*Processor or callable.*object",
    ):
        Sequential(cast(Any, object()))


def test_callable_has_parity_with_stateless_processor() -> None:
    table = _table()
    callable_pipeline = Sequential(_add_one, StandardScale())
    processor_pipeline = Sequential(_AddOne(), StandardScale())

    callable_output = callable_pipeline.fit_transform(table)
    processor_output = processor_pipeline.fit_transform(table)

    torch.testing.assert_close(
        callable_output.numerical,
        processor_output.numerical,
    )
    assert set(callable_pipeline.state_dict()) == set(
        processor_pipeline.state_dict()
    )

    for pipeline, output in (
        (callable_pipeline, callable_output),
        (processor_pipeline, processor_output),
    ):
        with pytest.raises(AttributeError, match="inverse_transform"):
            pipeline.inverse_transform(output)


def test_pipeline_passes_generator_to_steps() -> None:
    table = _table(torch.arange(200.0).view(100, 2))

    def _fit(seed: int) -> tuple[FeaturePermute, Quantile]:
        permute = FeaturePermute(method="random")
        quantile = Quantile(n_quantiles=6, subsample=32)
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
    assert repr(Sequential(StandardScale(), Power())) == dedent("""\
        Sequential(
          StandardScale(),
          Power(),
        )""")
    assert repr(Sequential(_add_one)) == dedent("""\
        Sequential(
          _add_one,
        )""")


def test_pipeline_checks_fitted_state() -> None:
    pipeline = Sequential(SoftmaxTemperature(), StandardScale())

    with pytest.raises(RuntimeError, match="'Sequential' is not fitted"):
        pipeline.transform(_table())


def test_pipeline_rejects_unsupported_stype() -> None:
    pipeline = Sequential(StandardScale())

    with pytest.raises(ValueError, match="categorical"):
        pipeline.fit_transform(_mixed_table())


def test_inverse_transform_rejects_non_invertible_step() -> None:
    processor = Sequential(MeanImpute())
    transformed = processor.fit_transform(_table())

    with pytest.raises(
        AttributeError,
        match=r"MeanImpute.*inverse_transform",
    ):
        processor.inverse_transform(transformed)


def test_inverse_transform_runs_steps_in_reverse_order() -> None:
    # Power and StandardScale do not commute, so the round trip only
    # reconstructs the input if the inverse applies the steps in reverse.
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
    )

    pipeline = Sequential(Power(), StandardScale())
    transformed = pipeline.fit_transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert torch.allclose(restored.numerical, table.numerical, atol=1e-4)
