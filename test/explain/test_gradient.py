from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch
from sdm import ColumnarTensor, Stype, TableTensor
from sdm.explain import (
    Explanation,
    ExplanationMode,
    FeatureAttribution,
    GradientSensitivity,
    InputSite,
    OutputIndex,
    UnsupportedExplanationError,
)
from torch import Tensor


def _table(
    values: list[list[float]],
    ids: list[int],
    *,
    columns: tuple[str, ...],
) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: columns,
            Stype.datetime: ("event_time",),
            Stype.id: ("user_id",),
        },
        numerical=torch.tensor(values),
        datetime=torch.arange(len(ids), dtype=torch.int64).unsqueeze(-1),
        id=ColumnarTensor((torch.tensor(ids),)),
    )


def _inputs() -> dict[InputSite, TableTensor]:
    return {
        InputSite(split="context"): _table(
            [[10.0], [20.0]],
            [1, 2],
            columns=("task_feature",),
        ),
        InputSite(split="context", table="users"): _table(
            [[1.0, 2.0], [3.0, 4.0]],
            [1, 2],
            columns=("value", "event_time__hour"),
        ),
        InputSite(split="query"): _table(
            [[0.0]],
            [3],
            columns=("task_feature",),
        ),
        InputSite(split="query", table="users"): _table(
            [[5.0, 6.0]],
            [3],
            columns=("value", "event_time__hour"),
        ),
        InputSite(split="query", table="unused"): _table(
            [[9.0]],
            [3],
            columns=("unused",),
        ),
    }


class _Evaluation:
    def __init__(self, *, zero: bool = False) -> None:
        self.inputs = _inputs()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))
        self.zero = zero
        self.num_calls = 0

    def __call__(
        self,
        replacements: Mapping[InputSite, Tensor] | None,
    ) -> TableTensor:
        self.num_calls += 1
        values = {
            site: (
                table.numerical
                if replacements is None or site not in replacements
                else replacements[site]
            )
            for site, table in self.inputs.items()
        }
        context = values[InputSite(split="context", table="users")]
        query = values[InputSite(split="query", table="users")]
        selected = (
            2 * context[0, 0]
            - 4 * context[1, 1]
            + 8 * query[0, 0]
            + 0 * query[0, 1]
        )
        if self.zero:
            selected = selected * 0
        other = 100 * query[0, 1] + self.bias
        return TableTensor.from_tensor(
            torch.stack((other, selected + self.bias)).reshape(1, 2),
            columns=("other", "prediction"),
        )


def _explain(
    method: GradientSensitivity,
    evaluation: _Evaluation,
) -> Explanation:
    with torch.inference_mode(False), torch.enable_grad():
        prediction = evaluation(None)
        return method.explain(
            evaluate=evaluation,
            inputs=evaluation.inputs,
            prediction=prediction,
            target=OutputIndex(row=0, column="prediction"),
            mode=ExplanationMode.full_context,
        )


def _attributions(
    explanation: Explanation,
) -> dict[InputSite, FeatureAttribution]:
    return {
        attribution.site: attribution
        for attribution in explanation.attributions
    }


def test_gradient_sensitivity_returns_signed_aligned_scores() -> None:
    evaluation = _Evaluation()
    existing_grad = torch.tensor(7.0)
    evaluation.bias.grad = existing_grad

    explanation = _explain(GradientSensitivity(), evaluation)

    torch.testing.assert_close(
        explanation.prediction.numerical,
        torch.tensor([[601.0, 27.0]]),
    )
    assert not explanation.prediction.numerical.requires_grad
    assert evaluation.bias.grad is existing_grad
    assert evaluation.num_calls == 2
    attributions = _attributions(explanation)
    assert attributions.keys() == evaluation.inputs.keys()

    context_site = InputSite(split="context", table="users")
    query_site = InputSite(split="query", table="users")
    torch.testing.assert_close(
        attributions[context_site].values.numerical,
        torch.tensor([[2.0, 0.0], [0.0, -4.0]]),
    )
    torch.testing.assert_close(
        attributions[query_site].values.numerical,
        torch.tensor([[8.0, 0.0]]),
    )

    for site, attribution in attributions.items():
        original = evaluation.inputs[site]
        assert attribution.score_kind == "gradient"
        assert attribution.input_space == "processed"
        assert attribution.signed
        assert attribution.normalization == "none"
        assert (
            attribution.values.columns[Stype.numerical]
            == (original.columns[Stype.numerical])
        )
        assert attribution.values.columns[Stype.datetime] == ()
        assert attribution.values.id is original.id

    for site in (
        InputSite(split="context"),
        InputSite(split="query"),
        InputSite(split="query", table="unused"),
    ):
        assert not attributions[site].values.numerical.any()


def test_gradient_sensitivity_magnitude_and_global_normalization() -> None:
    explanation = _explain(
        GradientSensitivity(
            magnitude=True,
            normalization="global_max_abs",
        ),
        _Evaluation(),
    )
    attributions = _attributions(explanation)
    context = attributions[InputSite(split="context", table="users")]
    query = attributions[InputSite(split="query", table="users")]

    assert not context.signed
    assert not query.signed
    assert context.normalization == query.normalization == "global_max_abs"
    torch.testing.assert_close(
        context.values.numerical,
        torch.tensor([[0.25, 0.0], [0.0, 0.5]]),
    )
    torch.testing.assert_close(
        query.values.numerical,
        torch.tensor([[1.0, 0.0]]),
    )


def test_gradient_sensitivity_normalization_preserves_sign() -> None:
    explanation = _explain(
        GradientSensitivity(normalization="global_max_abs"),
        _Evaluation(),
    )
    context = _attributions(explanation)[
        InputSite(split="context", table="users")
    ]

    assert context.signed
    torch.testing.assert_close(
        context.values.numerical,
        torch.tensor([[0.25, 0.0], [0.0, -0.5]]),
    )


def test_gradient_sensitivity_normalizes_all_zero_scores() -> None:
    explanation = _explain(
        GradientSensitivity(normalization="global_max_abs"),
        _Evaluation(zero=True),
    )

    for attribution in explanation.attributions:
        assert torch.isfinite(attribution.values.numerical).all()
        assert not attribution.values.numerical.any()


def test_gradient_sensitivity_rejects_disconnected_prediction() -> None:
    evaluation = _Evaluation()

    def disconnected(
        replacements: Mapping[InputSite, Tensor] | None,
    ) -> TableTensor:
        return TableTensor.from_tensor(
            torch.tensor([[1.0]]),
            columns=("prediction",),
        )

    prediction = disconnected(None)
    with pytest.raises(
        UnsupportedExplanationError,
        match="not differentiable",
    ):
        GradientSensitivity().explain(
            evaluate=disconnected,
            inputs=evaluation.inputs,
            prediction=prediction,
            target=OutputIndex(row=0, column=0),
            mode=ExplanationMode.full_context,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"magnitude": 1}, TypeError),
        ({"normalization": "per_table"}, ValueError),
    ],
)
def test_gradient_sensitivity_validates_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        GradientSensitivity(**cast(Any, kwargs))


def test_gradient_sensitivity_rejects_batched_inputs() -> None:
    evaluation = _Evaluation()
    site = InputSite(split="query")
    evaluation.inputs[site] = cast(
        TableTensor,
        evaluation.inputs[site].unsqueeze(0),
    )

    with pytest.raises(UnsupportedExplanationError, match="batched"):
        _explain(GradientSensitivity(), evaluation)


def test_feature_attribution_rejects_non_score_feature_blocks() -> None:
    with pytest.raises(ValueError, match="only contain numerical scores"):
        FeatureAttribution(
            site=InputSite(split="query"),
            values=_table([[1.0]], [1], columns=("value",)),
            score_kind="gradient",
            input_space="processed",
        )
