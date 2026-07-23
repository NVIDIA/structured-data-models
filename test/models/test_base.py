from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, cast

import pytest
import torch
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
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
    UnsupportedExplanationError,
)
from sdm.models import ICLModel
from sdm.processing import (
    EnsembleReduce,
    InvertibleMixin,
    Processor,
    Recipe,
    StandardScale,
    StypeDispatch,
)


@dataclass
class _Call:
    x_context: TableTensor | None
    x_query: TableTensor | None
    related_context_tables: RelatedTables | None
    related_query_tables: RelatedTables | None


class _RecordingModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[_Call] = []

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
        self.calls.append(
            _Call(
                x_context=x_context,
                x_query=x_query,
                related_context_tables=related_context_tables,
                related_query_tables=related_query_tables,
            )
        )
        table = x_query if x_query is not None else x_context
        assert table is not None
        return table.select_stypes(Stype.numerical)

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()


class _UnsupportedRecordingModel(_RecordingModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables = False


class _GeneratorRecordingProcessor(Processor, InvertibleMixin):
    supported_stypes = frozenset(Stype)
    generators: ClassVar[list[torch.Generator | None]] = []
    draws: ClassVar[list[torch.Tensor]] = []

    @classmethod
    def reset(cls) -> None:
        cls.generators.clear()
        cls.draws.clear()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.generators.append(generator)
        self.draws.append(torch.rand((), generator=generator))

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table


class _CountingStandardScale(StandardScale):
    fit_calls: ClassVar[int] = 0

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        type(self).fit_calls += 1
        super()._fit(table, generator=generator)


class _GradientModel(_RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(3.0))

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
        if kwargs.get("raise_error", False):
            raise RuntimeError("execution failed")
        out = super()._forward(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            cache=cache,
            generator=generator,
            **kwargs,
        )
        draw = torch.rand(
            (),
            device=out.device,
            generator=generator,
        )
        numerical = out.numerical
        source = x_query if x_query is not None else x_context
        assert source is not None
        if source.id.size(-1) > 0:
            # Exercise an ID-derived index that autograd saves for backward.
            index = source.id.unbind(-1)[0].long().mul(0)
            numerical = numerical.index_select(-2, index)
        if kwargs.get("branch_on_generator", False):
            numerical = numerical + (100 if generator is None else 200)
        if kwargs.get("nan_output", False):
            numerical = torch.full_like(numerical, torch.nan)
        return out.replace_blocks(numerical=numerical * self.weight + draw)


class _CallbackMethod(ExplanationMethod):
    def __init__(
        self,
        callback: Callable[
            [
                ExplanationCallable,
                ExplanationInputs,
                TableTensor,
                OutputIndex,
                ExplanationMode,
            ],
            Explanation,
        ],
        *,
        requirements: ExplanationRequirements | None = None,
    ) -> None:
        self.callback = callback
        self.requirements = (
            ExplanationRequirements(gradients=True)
            if requirements is None
            else requirements
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
        return self.callback(evaluate, inputs, prediction, target, mode)


def _result(
    method: ExplanationMethod,
    prediction: TableTensor,
    target: OutputIndex,
    mode: ExplanationMode,
) -> Explanation:
    prediction = prediction.replace_blocks(
        numerical=prediction.numerical.detach()
    )
    return Explanation(
        prediction=prediction,
        target=target.resolve(prediction),
        method=method.name,
        mode=mode,
    )


def _table(
    values: list[float],
    ids: list[int],
    *,
    value_column: str,
) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: (value_column,),
            Stype.id: ("user_id",),
        },
        numerical=torch.tensor(values).unsqueeze(-1),
        id=ColumnarTensor((torch.tensor(ids),)),
    )


def _related_tables(*, query: bool) -> RelatedTables:
    if query:
        users = _table([30.0], [3], value_column="age")
        orders = _table([106.0], [3], value_column="amount")
    else:
        users = _table([10.0, 20.0], [1, 2], value_column="age")
        orders = _table([100.0, 104.0], [1, 2], value_column="amount")

    return RelatedTables(
        tables={"users": users, "orders": orders},
        relationships=[
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            }
        ],
        task_links=[
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            }
        ],
    )


def _recipe() -> Recipe:
    return Recipe(
        features=StypeDispatch(numerical=StandardScale()),
    )


def _reduced_recipe() -> Recipe:
    return Recipe(
        features=StypeDispatch(numerical=StandardScale()),
        output=EnsembleReduce(),
    )


def _generator_recipe() -> Recipe:
    return Recipe(
        features=_GeneratorRecordingProcessor(),
        target=_GeneratorRecordingProcessor(),
    )


def _gradient_recipe() -> Recipe:
    return Recipe(
        features=StypeDispatch(numerical=_CountingStandardScale()),
        target=StandardScale(),
        output=EnsembleReduce(),
    )


def _gradient_inputs() -> tuple[TableTensor, TableTensor, TableTensor]:
    return (
        _table([0.0, 2.0], [1, 2], value_column="feature"),
        TableTensor.from_tensor(
            torch.tensor([[10.0], [14.0]]),
            columns=("target",),
        ),
        _table([3.0], [3], value_column="feature"),
    )


def _fit_draws(
    *,
    seed: int,
    cached: bool,
) -> list[torch.Tensor]:
    model = _RecordingModel()
    x_context = _table([0.0, 2.0], [1, 2], value_column="feature")
    y_context = TableTensor.from_tensor(torch.tensor([[0.0], [1.0]]))
    related_context = _related_tables(query=False)
    generator = torch.Generator().manual_seed(seed)

    _GeneratorRecordingProcessor.reset()
    if cached:
        model.fit(
            x_context,
            y_context,
            related_context,
            recipe=_generator_recipe(),
            num_estimators=2,
            generator=generator,
        )
    else:
        model(
            x_context,
            y_context,
            _table([3.0], [3], value_column="feature"),
            related_context,
            _related_tables(query=True),
            recipe=_generator_recipe(),
            num_estimators=2,
            generator=generator,
        )

    assert _GeneratorRecordingProcessor.generators == [generator] * 8
    return list(_GeneratorRecordingProcessor.draws)


@pytest.mark.parametrize("cached", [False, True])
def test_model_recipe_fitting_honors_generator(cached: bool) -> None:
    first = _fit_draws(seed=0, cached=cached)
    second = _fit_draws(seed=0, cached=cached)
    different_seed = _fit_draws(seed=1, cached=cached)

    assert len(first) == 8
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first, different_seed)
    )


@pytest.mark.parametrize("cached", [False, True])
def test_model_recipe_generator_does_not_advance_global_rng(
    cached: bool,
) -> None:
    state = torch.get_rng_state()

    _fit_draws(seed=0, cached=cached)

    assert torch.equal(torch.get_rng_state(), state)


def test_explain_full_context_is_differentiable_and_repeatable() -> None:
    model = _GradientModel().eval()
    x_context, y_context, x_query = _gradient_inputs()

    public_prediction = model(
        x_context,
        y_context,
        x_query,
        recipe=_gradient_recipe(),
        generator=torch.Generator().manual_seed(12),
    )
    assert public_prediction.numerical.is_inference()

    _CountingStandardScale.fit_calls = 0
    caller_generator = torch.Generator().manual_seed(12)
    generator_state = caller_generator.get_state().clone()
    parameter_grad = torch.tensor(7.0)
    model.weight.grad = parameter_grad.clone()
    raw_context = x_context.numerical.clone()
    raw_query = x_query.numerical.clone()

    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        endpoint: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        assert mode is ExplanationMode.full_context
        assert tuple(inputs) == (
            InputSite(split="context"),
            InputSite(split="query"),
        )
        assert all(not table.is_inference() for table in inputs.values())
        assert not endpoint.numerical.requires_grad
        query_site = InputSite(split="query")
        query = (
            inputs[query_site].numerical.detach().clone().requires_grad_(True)
        )
        prediction = evaluate({query_site: query})
        torch.testing.assert_close(
            prediction.numerical,
            public_prediction.numerical,
        )
        torch.testing.assert_close(prediction.numerical, endpoint.numerical)
        gradient = torch.autograd.grad(
            prediction.numerical.sum(),
            query,
        )[0]
        torch.testing.assert_close(gradient, torch.full_like(query, 6.0))

        with torch.no_grad():
            detached = evaluate(None)
        assert not detached.numerical.requires_grad

        changed = evaluate({query_site: query + 1})
        repeated = evaluate(None)
        torch.testing.assert_close(repeated.numerical, prediction.numerical)
        assert not changed.numerical.equal(repeated.numerical)
        return _result(method, endpoint, target, mode)

    method = _CallbackMethod(explain)
    with torch.inference_mode():
        explanation = model.explain_full_context(
            method,
            x_context,
            y_context,
            x_query,
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
            generator=caller_generator,
        )

    torch.testing.assert_close(
        explanation.prediction.numerical,
        public_prediction.numerical,
    )
    assert _CountingStandardScale.fit_calls == 1
    assert torch.equal(caller_generator.get_state(), generator_state)
    assert torch.equal(x_context.numerical, raw_context)
    assert torch.equal(x_query.numerical, raw_query)
    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(x_query)
    assert model.weight.grad is not None
    assert model.weight.grad.equal(parameter_grad)


def test_explain_full_context_normalizes_inference_inputs() -> None:
    model = _GradientModel().eval()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        endpoint: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        query_site = InputSite(split="query")
        assert not inputs[InputSite(split="context")].id.is_inference()
        assert not inputs[query_site].id.is_inference()
        query = (
            inputs[query_site].numerical.detach().clone().requires_grad_(True)
        )
        prediction = evaluate({query_site: query})
        gradient = torch.autograd.grad(
            prediction.numerical.sum(),
            query,
        )[0]
        torch.testing.assert_close(gradient, torch.full_like(query, 6.0))
        torch.testing.assert_close(prediction.numerical, endpoint.numerical)
        return _result(method, endpoint, target, mode)

    method = _CallbackMethod(explain)
    with torch.inference_mode():
        x_context, y_context, x_query = _gradient_inputs()
        model.explain_full_context(
            method,
            x_context,
            y_context,
            x_query,
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
            generator=torch.Generator().manual_seed(2),
        )


def test_explain_full_context_validates_method_and_replacements() -> None:
    model = _GradientModel().eval()
    x_context, y_context, x_query = _gradient_inputs()

    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        calls = len(model.calls)
        query_site = InputSite(split="query")
        query = inputs[query_site].numerical
        with torch.inference_mode():
            inference_query = query.clone()
        invalid = [
            ({InputSite(split="query", table="missing"): query}, KeyError),
            ({query_site: query.unsqueeze(0)}, ValueError),
            ({query_site: query.double()}, ValueError),
            ({query_site: inference_query}, ValueError),
        ]
        for replacements, error in invalid:
            with pytest.raises(error):
                evaluate(replacements)
            assert len(model.calls) == calls
        return _result(method, prediction, target, mode)

    method = _CallbackMethod(explain)
    model.train()
    with pytest.raises(UnsupportedExplanationError, match="evaluation mode"):
        model.explain_full_context(
            method,
            x_context,
            y_context,
            x_query,
            target=OutputIndex(row=0, column=0),
        )
    model.eval()

    with pytest.raises(UnsupportedExplanationError, match="one estimator"):
        model.explain_full_context(
            method,
            x_context,
            y_context,
            x_query,
            target=OutputIndex(row=0, column=0),
            num_estimators=2,
        )

    model.explain_full_context(
        method,
        x_context,
        y_context,
        x_query,
        target=OutputIndex(row=0, column=0),
        recipe=_gradient_recipe(),
    )


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        (
            ExplanationRequirements(
                supported_modes=frozenset({ExplanationMode.fitted})
            ),
            "does not support 'full_context'",
        ),
        (
            ExplanationRequirements(input_space="raw"),
            "does not yet expose 'raw'",
        ),
    ],
)
def test_explain_full_context_rejects_unsupported_requirements(
    requirements: ExplanationRequirements,
    message: str,
) -> None:
    method = _CallbackMethod(
        lambda evaluate, inputs, prediction, target, mode: cast(Any, None),
        requirements=requirements,
    )

    with pytest.raises(UnsupportedExplanationError, match=message):
        _GradientModel().eval().explain_full_context(
            method,
            *_gradient_inputs(),
            target=OutputIndex(row=0, column=0),
        )


def test_explain_full_context_requires_explanation_result() -> None:
    method = _CallbackMethod(
        lambda evaluate, inputs, prediction, target, mode: cast(
            Any, evaluate(None)
        )
    )

    with pytest.raises(TypeError, match="needs to return an 'Explanation'"):
        _GradientModel().eval().explain_full_context(
            method,
            *_gradient_inputs(),
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
        )


def test_explain_full_context_preserves_default_rng_and_ambient_modes() -> (
    None
):
    model = _GradientModel().eval()
    state = torch.get_rng_state().clone()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        first = evaluate(None)
        second = evaluate(None)
        torch.testing.assert_close(first.numerical, second.numerical)
        torch.testing.assert_close(first.numerical, prediction.numerical)
        return _result(method, prediction, target, mode)

    method = _CallbackMethod(explain)
    with torch.inference_mode():
        model.explain_full_context(
            method,
            *_gradient_inputs(),
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
        )
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()

    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()
    assert torch.equal(torch.get_rng_state(), state)


def test_explain_full_context_preserves_none_generator_semantics() -> None:
    model = _GradientModel().eval()
    state = torch.get_rng_state().clone()
    public_prediction = model(
        *_gradient_inputs(),
        recipe=_gradient_recipe(),
        branch_on_generator=True,
    )
    torch.set_rng_state(state)
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        repeated = evaluate(None)
        torch.testing.assert_close(
            prediction.numerical,
            public_prediction.numerical,
        )
        torch.testing.assert_close(
            repeated.numerical,
            public_prediction.numerical,
        )
        return _result(method, prediction, target, mode)

    method = _CallbackMethod(explain)
    explanation = model.explain_full_context(
        method,
        *_gradient_inputs(),
        target=OutputIndex(row=0, column=0),
        recipe=_gradient_recipe(),
        branch_on_generator=True,
    )

    torch.testing.assert_close(
        explanation.prediction.numerical,
        public_prediction.numerical,
    )
    assert torch.equal(torch.get_rng_state(), state)


def test_explain_full_context_accepts_matching_nan_prediction() -> None:
    model = _GradientModel().eval()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        return _result(method, prediction, target, mode)

    method = _CallbackMethod(explain)
    explanation = model.explain_full_context(
        method,
        *_gradient_inputs(),
        target=OutputIndex(row=0, column=0),
        recipe=_gradient_recipe(),
        nan_output=True,
    )

    assert explanation.prediction.numerical.isnan().all()


def test_explain_full_context_exception_restores_ambient_modes() -> None:
    model = _GradientModel().eval()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        return _result(method, prediction, target, mode)

    method = _CallbackMethod(explain)
    state = torch.get_rng_state().clone()
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="execution failed"):
            model.explain_full_context(
                method,
                *_gradient_inputs(),
                target=OutputIndex(row=0, column=0),
                recipe=_gradient_recipe(),
                raise_error=True,
            )
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()

    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()
    assert torch.equal(torch.get_rng_state(), state)


def test_explain_full_context_related_replacement_is_call_local() -> None:
    model = _RecordingModel().eval()
    related_context = _related_tables(query=False)
    related_query = _related_tables(query=True)
    raw_query = related_query.tables["users"].numerical.clone()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        endpoint: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        assert tuple(inputs) == (
            InputSite(split="context"),
            InputSite(split="context", table="users"),
            InputSite(split="context", table="orders"),
            InputSite(split="query"),
            InputSite(split="query", table="users"),
            InputSite(split="query", table="orders"),
        )
        site = InputSite(split="query", table="users")
        original = inputs[site].numerical.clone()
        evaluate({site: original + 1})

        call = model.calls[-1]
        assert call.related_query_tables is not None
        table = call.related_query_tables.tables["users"]
        torch.testing.assert_close(table.numerical, original + 1)
        assert call.related_query_tables.relationships == (
            related_query.relationships
        )
        assert call.related_query_tables.task_links == related_query.task_links
        assert table.id.tolist() == related_query.tables["users"].id.tolist()
        assert table.schema == inputs[site].schema
        torch.testing.assert_close(inputs[site].numerical, original)
        return _result(method, endpoint, target, mode)

    method = _CallbackMethod(explain)
    model.explain_full_context(
        method,
        _table([0.0, 2.0], [1, 2], value_column="feature"),
        TableTensor.from_tensor(torch.tensor([[0.0], [1.0]])),
        _table([3.0], [3], value_column="feature"),
        related_context,
        related_query,
        target=OutputIndex(row=0, column=0),
        recipe=_reduced_recipe(),
    )

    torch.testing.assert_close(
        related_query.tables["users"].numerical,
        raw_query,
    )


def test_explain_full_context_rejects_non_endpoint_prediction() -> None:
    model = _GradientModel().eval()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        query_site = InputSite(split="query")
        changed = evaluate({query_site: inputs[query_site].numerical + 1})
        return _result(method, changed, target, mode)

    method = _CallbackMethod(explain)
    with pytest.raises(ValueError, match="unmodified model prediction"):
        model.explain_full_context(
            method,
            *_gradient_inputs(),
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
        )


def test_explain_full_context_validates_attribution_alignment() -> None:
    model = _GradientModel().eval()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        result = _result(method, prediction, target, mode)
        query_site = InputSite(split="query")
        scores = inputs[query_site].select_stypes((Stype.numerical, Stype.id))
        scores = scores.replace_blocks(
            numerical=scores.numerical.detach(),
            id=ColumnarTensor((torch.tensor([999]),)),
        )
        return replace(
            result,
            attributions=(
                FeatureAttribution(
                    site=query_site,
                    values=scores,
                    score_kind="test",
                    input_space="processed",
                ),
            ),
        )

    method = _CallbackMethod(explain)
    with pytest.raises(ValueError, match="identifiers do not match"):
        model.explain_full_context(
            method,
            *_gradient_inputs(),
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
        )


def test_explain_full_context_rejects_unexposed_attribution_site() -> None:
    model = _GradientModel().eval()
    method: _CallbackMethod

    def explain(
        evaluate: ExplanationCallable,
        inputs: ExplanationInputs,
        prediction: TableTensor,
        target: OutputIndex,
        mode: ExplanationMode,
    ) -> Explanation:
        result = _result(method, prediction, target, mode)
        values = next(iter(inputs.values())).select_stypes(
            (Stype.numerical, Stype.id)
        )
        values = values.replace_blocks(numerical=values.numerical.detach())
        return replace(
            result,
            attributions=(
                FeatureAttribution(
                    site=InputSite(split="query", table="missing"),
                    values=values,
                    score_kind="test",
                    input_space="processed",
                ),
            ),
        )

    method = _CallbackMethod(explain)
    with pytest.raises(ValueError, match="was not exposed"):
        model.explain_full_context(
            method,
            *_gradient_inputs(),
            target=OutputIndex(row=0, column=0),
            recipe=_gradient_recipe(),
        )


def test_related_table_preprocessing_forward_and_cache() -> None:
    model = _RecordingModel()
    x_context = _table([0.0, 2.0], [1, 2], value_column="feature")
    x_query = _table([3.0], [3], value_column="feature")
    y_context = TableTensor.from_tensor(torch.tensor([[0.0], [1.0]]))
    related_context = _related_tables(query=False)
    full_related_query = _related_tables(query=True)
    related_query = RelatedTables(
        tables={
            "users": full_related_query.tables["users"],
            "events": _table([9.0], [3], value_column="event_value"),
        },
        relationships=(),
        task_links=full_related_query.task_links,
    )

    direct = cast(
        TableTensor,
        model(
            x_context,
            y_context,
            x_query,
            related_context,
            related_query,
            recipe=_recipe(),
            num_estimators=2,
        ),
    )

    assert len(model.calls) == 2
    call = model.calls[0]
    assert call.x_context is not None
    assert call.x_query is not None
    assert call.related_context_tables is not None
    assert call.related_query_tables is not None
    assert set(call.related_query_tables.tables) == {"users"}
    torch.testing.assert_close(
        call.x_context.numerical,
        torch.tensor([[-1.0], [1.0]]),
    )
    torch.testing.assert_close(call.x_query.numerical, torch.tensor([[2.0]]))
    torch.testing.assert_close(
        call.related_query_tables.tables["users"].numerical,
        torch.tensor([[3.0]]),
    )
    assert call.x_context.id.tolist() == x_context.id.tolist()
    assert call.x_query.id.tolist() == x_query.id.tolist()
    for name in related_context.tables:
        assert (
            call.related_context_tables.tables[name].id.tolist()
            == related_context.tables[name].id.tolist()
        )
    for name in call.related_query_tables.tables:
        assert (
            call.related_query_tables.tables[name].id.tolist()
            == related_query.tables[name].id.tolist()
        )
    assert (
        call.related_context_tables.relationships
        == related_context.relationships
    )
    assert call.related_context_tables.task_links == related_context.task_links
    assert (
        call.related_query_tables.relationships == related_query.relationships
    )
    assert call.related_query_tables.task_links == related_query.task_links

    model.calls.clear()
    model.fit(
        x_context,
        y_context,
        related_context,
        recipe=_recipe(),
        num_estimators=2,
    )
    assert model._caches is not None
    processors = [
        cast(dict[str, Processor], cache["related_processors"])
        for cache in model._caches
    ]
    assert (
        len(
            {
                id(processor)
                for estimator in processors
                for processor in estimator.values()
            }
        )
        == 4
    )

    prediction = model.predict(x_query, related_query)

    torch.testing.assert_close(prediction.numerical, direct.numerical)
    assert len(model.calls) == 4
    assert model.calls[0].related_context_tables is not None
    assert model.calls[0].related_query_tables is None
    assert model.calls[-1].related_context_tables is None
    assert model.calls[-1].related_query_tables is not None
    torch.testing.assert_close(
        model.calls[-1].related_query_tables.tables["users"].numerical,
        torch.tensor([[3.0]]),
    )


def test_model_input_validation() -> None:
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)

    with pytest.raises(ValueError, match="one column"):
        model(x_context, torch.randn(4, 2), x_query)
    with pytest.raises(ValueError, match="matching row dimensions"):
        model(x_context, torch.randn(3, 1), x_query)
    with pytest.raises(ValueError, match="same schema"):
        model(x_context, y_context, torch.randn(2, 4))


def test_predict_validates_cached_input_schema() -> None:
    model = _RecordingModel()
    model.fit(torch.randn(2, 4, 3), torch.randn(2, 4, 1))

    with pytest.raises(ValueError, match="same schema"):
        model.predict(torch.randn(2, 3, 4))


def test_related_table_validation() -> None:
    x_context = _table([0.0, 2.0], [1, 2], value_column="feature")
    x_query = _table([3.0], [3], value_column="feature")
    y_context = TableTensor.from_tensor(torch.tensor([[0.0], [1.0]]))
    related_context = _related_tables(query=False)
    related_query = _related_tables(query=True)

    unsupported_model = _UnsupportedRecordingModel()
    with pytest.raises(ValueError, match="related tables"):
        unsupported_model(
            x_context,
            y_context,
            x_query,
            related_context,
            related_query,
        )
    with pytest.raises(ValueError, match="does not support related tables"):
        unsupported_model.fit(x_context, y_context, related_context)
    unsupported_model.fit(x_context, y_context)
    with pytest.raises(ValueError, match="related tables to be provided"):
        unsupported_model.predict(x_query, related_query)

    model = _RecordingModel()
    model.fit(x_context, y_context, related_context, recipe=_recipe())
    mismatched_query = RelatedTables(
        tables={"users": _table([30.0], [3], value_column="different_column")},
        relationships=related_query.relationships,
        task_links=related_query.task_links,
    )
    with pytest.raises(ValueError, match="share the same schema"):
        model.predict(x_query, mismatched_query)
