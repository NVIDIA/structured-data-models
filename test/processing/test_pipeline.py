from typing import cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    InvertibleMixin,
    MeanImpute,
    Pipeline,
    Power,
    Processor,
    SoftmaxTemperature,
    StandardScale,
)


class Add(Processor):
    requires_fit = False

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.value


class ReverseBlocks(Processor, InvertibleMixin):
    requires_fit = False
    input_scope = "table"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(input, TableTensor):
            raise TypeError("Expected TableTensor")

        numerical_index = torch.arange(
            input.numerical.size(-1) - 1,
            -1,
            -1,
            device=input.numerical.device,
        )
        categorical_index = torch.arange(
            input.categorical.size(-1) - 1,
            -1,
            -1,
            device=input.categorical.device,
        )
        return TableTensor(
            columns={
                Stype.numerical: tuple(
                    reversed(input.columns[Stype.numerical])
                ),
                Stype.categorical: tuple(
                    reversed(input.columns[Stype.categorical])
                ),
            },
            numerical=input.numerical.index_select(-1, numerical_index),
            categorical=cast(
                CategoricalTensor,
                input.categorical.index_select(-1, categorical_index),
            ),
        )

    def _inverse_transform(self, input: torch.Tensor) -> torch.Tensor:
        return self.forward(input)


def _wide_table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("x0", "x1", "x2"),
            "categorical": ("kind", "segment"),
        },
        numerical=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["a", "b"]),
                StringTensor.from_list(["small", "large"]),
            ),
        ),
    )


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
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


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Pipeline().transform(table) is table
    assert Pipeline().fit_transform(table) is table
    assert Pipeline().inverse_transform(table) is table


def test_pipeline_transforms_numerical_and_passes_categorical() -> None:
    table = _table()

    output = Pipeline([StandardScale()]).fit_transform(table)

    assert not torch.equal(output.numerical, table.numerical)
    assert output.categorical is table.categorical
    assert output.columns == table.columns


def test_repr_lists_steps_or_reports_identity() -> None:
    assert repr(Pipeline()) == "Pipeline(identity)"
    assert repr(Pipeline([StandardScale(), Power()])) == (
        "Pipeline(StandardScale -> Power)"
    )


def test_pipeline_rejects_non_processor_step() -> None:
    with pytest.raises(TypeError, match="Expected a Processor step"):
        Pipeline([object()])  # ty: ignore[invalid-argument-type]


def test_pipeline_error_includes_step_position() -> None:
    # The second step is unfitted, so its transform raises with its position.
    pipeline = Pipeline([SoftmaxTemperature(), StandardScale()])

    with pytest.raises(RuntimeError, match=r"step 1 \(StandardScale\)"):
        pipeline.transform(_table())


def test_inverse_transform_rejects_non_invertible_step() -> None:
    # MeanImpute is not invertible, so inverse_transform reports its position.
    with pytest.raises(TypeError, match=r"step 0 \(MeanImpute\)"):
        Pipeline([MeanImpute()]).inverse_transform(_table())


def test_inverse_transform_runs_steps_in_reverse_order() -> None:
    # Power and StandardScale do not commute, so the round trip only
    # reconstructs the input if the inverse applies the steps in reverse.
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
    )

    pipeline = Pipeline([Power(), StandardScale()])
    transformed = pipeline.fit_transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert torch.allclose(restored.numerical, table.numerical, atol=1e-4)


def test_table_processor_composes_with_block_processor() -> None:
    table = _wide_table()

    output = Pipeline([ReverseBlocks(), Add(10)]).transform(table)

    assert output.columns[Stype.numerical] == ("x2", "x1", "x0")
    assert output.columns[Stype.categorical] == ("segment", "kind")
    assert torch.equal(
        output.numerical,
        table.numerical.index_select(-1, torch.tensor([2, 1, 0])) + 10,
    )
    assert torch.equal(
        output.categorical.as_tensor(),
        table.categorical.as_tensor().index_select(-1, torch.tensor([1, 0])),
    )


def test_fit_threads_table_processor_output_to_later_steps() -> None:
    table = _wide_table()
    scale = StandardScale()

    Pipeline([ReverseBlocks(), scale]).fit(table)

    reversed_numerical = table.numerical.index_select(
        -1, torch.tensor([2, 1, 0])
    )
    assert torch.equal(scale.mean, reversed_numerical.mean(dim=0))


def test_table_processor_inverse_restores_blocks() -> None:
    table = _wide_table()
    pipeline = Pipeline([ReverseBlocks()])

    transformed = pipeline.transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert restored.columns == table.columns
    assert torch.equal(restored.numerical, table.numerical)
    assert torch.equal(
        restored.categorical.as_tensor(), table.categorical.as_tensor()
    )


def test_table_processor_bad_output_reports_step_position() -> None:
    class BadTableOutput(Processor):
        requires_fit = False
        input_scope = "table"

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.empty(0)

    with pytest.raises(TypeError, match=r"step 0 \(BadTableOutput\)"):
        Pipeline([BadTableOutput()]).transform(_table())
