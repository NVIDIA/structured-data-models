from typing import Any, cast

import pytest
import torch
from sdm import TableTensor
from sdm.explain import (
    Explanation,
    ExplanationCallable,
    ExplanationInputs,
    ExplanationMethod,
    ExplanationMode,
    ExplanationRequirements,
    FeatureAttribution,
    InputSite,
    OutputIndex,
    ResolvedTarget,
)


class _Method(ExplanationMethod):
    requirements = ExplanationRequirements(
        gradients=True,
        supported_modes=frozenset({ExplanationMode.full_context}),
    )

    def explain(
        self,
        *,
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        return Explanation(
            prediction=prediction,
            target=target.resolve(prediction),
            method=self.name,
            mode=mode,
        )


def _prediction() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        columns=("negative", "positive"),
    )


def test_method_contract_is_a_normal_abstract_base_class() -> None:
    method = _Method()

    assert isinstance(method, ExplanationMethod)
    assert method.name == "_Method"
    assert method.requirements.gradients
    assert method.requirements.supported_modes == frozenset(
        {ExplanationMode.full_context}
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (OutputIndex(row=2, column=0), "outside the prediction rows"),
        (
            OutputIndex(row=0, column=2),
            "outside the numerical prediction columns",
        ),
        (
            OutputIndex(row=0, column="missing"),
            "Unknown numerical prediction column",
        ),
    ],
)
def test_output_index_rejects_out_of_bounds(
    target: OutputIndex,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        target.resolve(_prediction())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"row": -1, "column": 0}, "'row'"),
        ({"row": 0, "column": -1}, "Integer 'column'"),
        ({"row": True, "column": 0}, "'row'"),
        ({"row": 0, "column": False}, "'column'"),
        ({"row": 1.5, "column": 0}, "'row'"),
        ({"row": 0, "column": 1.5}, "'column'"),
        ({"row": 0, "column": ""}, "String 'column'"),
    ],
)
def test_output_index_rejects_negative_and_boolean_indices(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OutputIndex(**cast(Any, kwargs))


def test_output_index_rejects_batched_prediction() -> None:
    prediction = TableTensor.from_tensor(
        torch.rand(2, 3, 4),
        columns=("a", "b", "c", "d"),
    )

    with pytest.raises(ValueError, match="unbatched"):
        OutputIndex(row=0, column=0).resolve(prediction)


def test_explanation_contains_table_tensor_attributions() -> None:
    prediction = _prediction()
    values = TableTensor.from_tensor(
        torch.tensor([[0.2, -0.3]]),
        columns=("age", "income"),
    )
    attribution = FeatureAttribution(
        site=InputSite(split="query"),
        values=values,
        score_kind="integrated_gradients",
        input_space="processed",
    )

    explanation = Explanation(
        prediction=prediction,
        target=OutputIndex(row=1, column="positive").resolve(prediction),
        method="integrated_gradients",
        mode=ExplanationMode.full_context,
        attributions=(attribution,),
    )

    assert explanation.prediction is prediction
    assert explanation.attributions[0].values is values
    assert explanation.attributions[0].site == InputSite(split="query")


def test_explanation_requires_tuple_attributions() -> None:
    prediction = _prediction()

    with pytest.raises(TypeError, match="tuple"):
        Explanation(
            prediction=prediction,
            target=OutputIndex(row=0, column=0).resolve(prediction),
            method="method",
            mode=ExplanationMode.full_context,
            attributions=cast(tuple, []),
        )


def test_explanation_rejects_target_that_does_not_match_prediction() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Explanation(
            prediction=_prediction(),
            target=ResolvedTarget(
                row=0,
                column=1,
                column_name="wrong",
            ),
            method="method",
            mode=ExplanationMode.full_context,
        )


def test_resolved_target_rejects_unknown_output_space() -> None:
    with pytest.raises(ValueError, match="'prediction'"):
        ResolvedTarget(
            row=0,
            column=0,
            column_name="negative",
            output_space=cast(Any, "logit"),
        )
