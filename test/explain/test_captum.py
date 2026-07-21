import builtins
import importlib.util
import subprocess
import sys
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import pytest
import torch
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.explain import (
    CaptumIntegratedGradients,
    ExplanationMode,
    FeatureAttribution,
    InputSite,
    IntegratedGradientsDiagnostics,
    OutputIndex,
    UnsupportedExplanationError,
)
from sdm.models import ICLModel
from sdm.processing import EnsembleReduce, Recipe

requires_captum = pytest.mark.skipif(
    importlib.util.find_spec("captum") is None,
    reason="Captum is not installed",
)


class _LinearICLModel(ICLModel):
    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_related_tables: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))
        self.grad_modes: list[bool] = []
        self.eval()

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        assert x_context is not None
        assert x_query is not None
        self.grad_modes.append(torch.is_grad_enabled())
        selected = (
            2 * x_context.numerical[0, 0]
            - 4 * x_context.numerical[1, 1]
            + 8 * x_query.numerical[0, 0]
            + 0 * x_query.numerical[0, 1]
            + self.bias
        )
        other = 100 * x_query.numerical[0, 1] + self.bias
        return TableTensor.from_tensor(
            torch.stack((other, selected)).reshape(1, 2),
            columns=("other", "prediction"),
        )

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe(output=EnsembleReduce())


def _model_inputs() -> tuple[TableTensor, TableTensor, TableTensor]:
    return (
        TableTensor(
            columns={
                Stype.numerical: ("a", "b"),
                Stype.id: ("row_id",),
            },
            numerical=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            id=ColumnarTensor((torch.tensor([1, 2]),)),
        ),
        TableTensor.from_tensor(
            torch.tensor([[0.0], [1.0]]),
            columns=("target",),
        ),
        TableTensor(
            columns={
                Stype.numerical: ("a", "b"),
                Stype.id: ("row_id",),
            },
            numerical=torch.tensor([[5.0, 6.0]]),
            id=ColumnarTensor((torch.tensor([3]),)),
        ),
    )


def _attributions(
    values: tuple[FeatureAttribution, ...],
) -> dict[InputSite, FeatureAttribution]:
    return {attribution.site: attribution for attribution in values}


@requires_captum
def test_captum_integrated_gradients_returns_analytic_attributions() -> None:
    model = _LinearICLModel()
    args = _model_inputs()
    expected = model(*args)
    model.grad_modes.clear()
    existing_grad = torch.tensor(9.0)
    model.bias.grad = existing_grad

    with torch.inference_mode():
        explanation = model.explain_full_context(
            CaptumIntegratedGradients(n_steps=16),
            *args,
            target=OutputIndex(row=0, column="prediction"),
        )
        assert torch.is_inference_mode_enabled()

    assert explanation.mode is ExplanationMode.full_context
    assert explanation.prediction.allclose(expected)
    assert not explanation.prediction.numerical.is_inference()
    assert not explanation.prediction.numerical.requires_grad
    assert model.bias.grad is existing_grad
    assert model.grad_modes[:3] == [True, False, False]
    assert any(model.grad_modes[3:])

    attributions = _attributions(explanation.attributions)
    assert attributions.keys() == {
        InputSite(split="context"),
        InputSite(split="query"),
    }
    torch.testing.assert_close(
        attributions[InputSite(split="context")].values.numerical,
        torch.tensor([[2.0, 0.0], [0.0, -16.0]]),
    )
    torch.testing.assert_close(
        attributions[InputSite(split="query")].values.numerical,
        torch.tensor([[40.0, 0.0]]),
    )
    for attribution in explanation.attributions:
        assert attribution.score_kind == "integrated_gradients"
        assert attribution.input_space == "processed"
        assert attribution.signed
        assert attribution.normalization == "none"

    assert len(explanation.diagnostics) == 1
    diagnostics = explanation.diagnostics[0]
    assert isinstance(diagnostics, IntegratedGradientsDiagnostics)
    assert diagnostics.varied_sites == (
        InputSite(split="context"),
        InputSite(split="query"),
    )
    assert diagnostics.n_steps == 16
    torch.testing.assert_close(
        diagnostics.reference_prediction.numerical,
        torch.tensor([[1.0, 1.0]]),
    )
    assert diagnostics.convergence_delta.item() == pytest.approx(
        0.0,
        abs=1e-5,
    )
    assert sum(
        attribution.values.numerical.sum().item()
        for attribution in explanation.attributions
    ) == pytest.approx(26.0)


@requires_captum
def test_captum_partial_baseline_varies_only_selected_sites() -> None:
    query_site = InputSite(split="query")
    baseline = TableTensor(
        columns={
            Stype.numerical: ("a", "b"),
            Stype.id: ("row_id",),
        },
        numerical=torch.tensor([[1.0, 2.0]]),
        id=ColumnarTensor((torch.tensor([3]),)),
    )

    explanation = _LinearICLModel().explain_full_context(
        CaptumIntegratedGradients(
            {query_site: baseline},
            n_steps=8,
        ),
        *_model_inputs(),
        target=OutputIndex(row=0, column="prediction"),
    )

    assert tuple(
        attribution.site for attribution in explanation.attributions
    ) == (query_site,)
    torch.testing.assert_close(
        explanation.attributions[0].values.numerical,
        torch.tensor([[32.0, 0.0]]),
    )
    diagnostics = explanation.diagnostics[0]
    assert isinstance(diagnostics, IntegratedGradientsDiagnostics)
    assert diagnostics.varied_sites == (query_site,)
    torch.testing.assert_close(
        diagnostics.reference_prediction.numerical,
        torch.tensor([[201.0, -5.0]]),
    )


@requires_captum
def test_captum_scalar_baseline_selects_one_site() -> None:
    query_site = InputSite(split="query")

    explanation = _LinearICLModel().explain_full_context(
        CaptumIntegratedGradients({query_site: 0.0}, n_steps=8),
        *_model_inputs(),
        target=OutputIndex(row=0, column="prediction"),
    )

    assert tuple(
        attribution.site for attribution in explanation.attributions
    ) == (query_site,)
    torch.testing.assert_close(
        explanation.attributions[0].values.numerical,
        torch.tensor([[40.0, 0.0]]),
    )


def test_captum_dependency_is_loaded_only_when_explaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        global_vars: Mapping[str, object] | None = None,
        local_vars: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "captum.attr":
            raise ModuleNotFoundError(
                "No module named 'captum'",
                name="captum",
            )
        return original_import(
            name,
            global_vars,
            local_vars,
            fromlist,
            level,
        )

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(
        ImportError,
        match=r"structured-data-models\[captum\]",
    ):
        _LinearICLModel().explain_full_context(
            CaptumIntegratedGradients(),
            *_model_inputs(),
            target=OutputIndex(row=0, column="prediction"),
        )


def test_importing_explain_does_not_import_captum() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import sdm.explain; "
                "assert not any(name == 'captum' or "
                "name.startswith('captum.') for name in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"baselines": object()}, TypeError),
        ({"baselines": {}}, ValueError),
        ({"baselines": {"query": torch.tensor(0.0)}}, TypeError),
        ({"baselines": {InputSite("query"): object()}}, TypeError),
        ({"baselines": {InputSite("query"): True}}, TypeError),
        ({"n_steps": 0}, ValueError),
        ({"n_steps": True}, ValueError),
    ],
)
def test_captum_integrated_gradients_validates_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CaptumIntegratedGradients(**cast(Any, kwargs))


@requires_captum
@pytest.mark.parametrize(
    ("baselines", "message"),
    [
        (
            {InputSite(split="query", table="users"): torch.zeros(1, 2)},
            "Unknown Captum baseline",
        ),
        (
            {InputSite(split="query"): torch.zeros(2, 2)},
            "needs shape",
        ),
        (
            {InputSite(split="query"): torch.zeros(1, 2).double()},
            "needs dtype",
        ),
        (
            {
                InputSite(split="query"): TableTensor.from_tensor(
                    torch.zeros(1, 2),
                    columns=("renamed", "b"),
                )
            },
            "same numerical columns",
        ),
        (
            {
                InputSite(split="query"): TableTensor(
                    columns={
                        Stype.numerical: ("a", "b"),
                        Stype.id: ("row_id",),
                    },
                    numerical=torch.zeros(1, 2),
                    id=ColumnarTensor((torch.tensor([99]),)),
                )
            },
            "same identifiers",
        ),
    ],
)
def test_captum_integrated_gradients_validates_baseline_at_execution(
    baselines: Mapping[InputSite, torch.Tensor],
    message: str,
) -> None:
    with pytest.raises(
        (ValueError, UnsupportedExplanationError),
        match=message,
    ):
        _LinearICLModel().explain_full_context(
            CaptumIntegratedGradients(baselines),
            *_model_inputs(),
            target=OutputIndex(row=0, column="prediction"),
        )


def test_captum_integrated_gradients_rejects_fitted_execution() -> None:
    model = _LinearICLModel()
    _, _, x_query = _model_inputs()

    with pytest.raises(
        UnsupportedExplanationError,
        match="does not support 'fitted'",
    ):
        model.explain_fitted(
            CaptumIntegratedGradients(),
            x_query,
            target=OutputIndex(row=0, column="prediction"),
        )
