import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    MeanImpute,
    Power,
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


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Sequential().transform(table) is table
    assert Sequential().fit_transform(table) is table
    assert Sequential().inverse_transform(table) is table


def test_pipeline_transforms_numerical() -> None:
    table = _table()

    output = Sequential(StandardScale()).fit_transform(table)

    assert not torch.equal(output.numerical, table.numerical)


def test_repr_lists_steps() -> None:
    assert repr(Sequential()) == "Sequential()"
    assert repr(Sequential(StandardScale(), Power())) == (
        "Sequential(\n  StandardScale(),\n  Power(),\n)"
    )


def test_pipeline_checks_step_fitted_state() -> None:
    pipeline = Sequential(SoftmaxTemperature(), StandardScale())

    with pytest.raises(RuntimeError, match="'StandardScale' is not fitted"):
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
