import hashlib
import json
import pickle
from typing import Any, ClassVar, cast

import pytest
import torch

import sdm.processing as sp
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.models import EnsembleParallel, ICLModel
from sdm.processing import InvertibleMixin, Processor


class _WorkerModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables: ClassVar[bool] = True

    def __init__(self, worker_id: int) -> None:
        super().__init__()
        self.register_buffer("worker_id", torch.tensor(worker_id))
        self.seen_num_hops: list[int | None] = []
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
        worker_id = cast(torch.Tensor, self.worker_id)
        num_hops = cast(int | None, kwargs.get("num_hops"))
        self.seen_num_hops.append(num_hops)
        if cache is not None and cache.is_recording:
            cache["worker_id"] = worker_id.clone()
            cache["num_hops"] = num_hops
        elif cache is not None:
            torch.testing.assert_close(
                cast(torch.Tensor, cache["worker_id"]),
                worker_id,
            )
            assert cache["num_hops"] == num_hops

        table = x_query if x_query is not None else x_context
        assert table is not None
        numerical = table.numerical[..., :1] + 100 * worker_id
        if related_query_tables is not None:
            numerical = (
                numerical
                + related_query_tables.tables["users"].numerical[..., :1]
            )
        return TableTensor(
            columns={Stype.numerical: ("prediction",)},
            numerical=numerical,
        )

    @classmethod
    def default_recipe(cls) -> sp.Recipe:
        return sp.Recipe()


class _CountOutput(Processor):
    handles_stypes = frozenset(Stype)
    requires_fit = False
    calls: ClassVar[int] = 0

    def _transform(self, table: TableTensor) -> TableTensor:
        type(self).calls += 1
        return table


class _GeneratorProcessor(Processor, InvertibleMixin):
    handles_stypes = frozenset(Stype)
    requires_fit = True
    draws: ClassVar[list[torch.Tensor]] = []

    @classmethod
    def reset(cls) -> None:
        cls.draws.clear()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.draws.append(torch.rand((), generator=generator))

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table


class _RandomModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical})
    supports_related_tables = False

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("marker", torch.tensor(0.0))
        self.calls = 0
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
        self.calls += 1
        table = x_query if x_query is not None else x_context
        assert table is not None
        if cache is not None and cache.is_replaying:
            draw = cast(torch.Tensor, cache["draw"])
        else:
            draw = torch.rand(
                (2,),
                device=table.device,
                generator=generator,
            )
            if cache is not None:
                cache["draw"] = draw
        return TableTensor(
            columns={Stype.numerical: ("prediction",)},
            numerical=table.numerical[..., :1] + draw.sum(),
        )

    @classmethod
    def default_recipe(cls) -> sp.Recipe:
        return sp.Recipe()


class _ReplacementRandomModel(_RandomModel):
    pass


class _FitMutatingRandomModel(_RandomModel):
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
        if cache is not None and cache.is_recording:
            cast(torch.Tensor, self.marker).add_(1)
        return super()._forward(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            cache=cache,
            generator=generator,
            **kwargs,
        )


class _FailModel(_WorkerModel):
    def __init__(self, worker_id: int, *, fail_fit: bool = False) -> None:
        super().__init__(worker_id)
        self.fail_fit = fail_fit
        self.fail_predict = False

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
        if cache is not None and cache.is_recording and self.fail_fit:
            raise ValueError("fit sentinel")
        if cache is not None and cache.is_replaying and self.fail_predict:
            raise ValueError("predict sentinel")
        return super()._forward(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            cache=cache,
            generator=generator,
            **kwargs,
        )


class _GradientCallback(Callback):
    requires_grad = True


class _CountingCallback(Callback):
    def __init__(self) -> None:
        self.starts = 0

    def on_forward_start(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.starts += 1


def _table(values: list[float]) -> TableTensor:
    return TableTensor.from_tensor(torch.tensor(values).unsqueeze(-1))


def _random_recipe(*, reduce: bool = False) -> sp.Recipe:
    return sp.Recipe(
        features=_GeneratorProcessor(),
        target=_GeneratorProcessor(),
        output=sp.ReduceEstimators() if reduce else sp.Identity(),
    )


def _assign_attribute(obj: object, name: str, value: object) -> None:
    setattr(obj, name, value)


def _id_table(values: list[float], ids: list[int]) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: ("user_id",),
        },
        numerical=torch.tensor(values).unsqueeze(-1),
        id=ColumnarTensor((torch.tensor(ids),)),
    )


def _related(values: list[float], ids: list[int]) -> RelatedTables:
    return RelatedTables(
        tables={"users": _id_table(values, ids)},
        relationships=(),
        task_links=(
            {
                "task_column": "user_id",
                "table": "users",
                "table_column": "user_id",
            },
        ),
    )


def test_stable_estimator_order_and_single_output_finalization() -> None:
    models = [_WorkerModel(0), _WorkerModel(1)]
    runner = EnsembleParallel(models, rng_policy="member")
    recipe = sp.Recipe(output=_CountOutput())
    _CountOutput.calls = 0

    output = runner(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        _table([2.0]),
        recipe=recipe,
        num_estimators=5,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[[2.0]], [[102.0]], [[2.0]], [[102.0]], [[2.0]]]),
    )
    assert _CountOutput.calls == 1
    assert output.device == torch.device("cpu")
    runner.close()


def test_single_replica_matches_serial_forward_and_cache() -> None:
    serial = _WorkerModel(0)
    replica = _WorkerModel(0)
    runner = EnsembleParallel([replica])
    x_context = _table([0.0, 1.0])
    y_context = _table([0.0, 1.0])
    x_query = _table([2.0, 3.0])

    expected = serial(
        x_context,
        y_context,
        x_query,
        num_estimators=3,
    )
    actual = runner(
        x_context,
        y_context,
        x_query,
        num_estimators=3,
    )
    torch.testing.assert_close(actual.numerical, expected.numerical)

    serial.fit(x_context, y_context, num_estimators=3)
    runner.fit(x_context, y_context, num_estimators=3)
    torch.testing.assert_close(
        runner.predict(x_query).numerical,
        serial.predict(x_query).numerical,
    )
    torch.testing.assert_close(
        runner.predict(_table([4.0])).numerical,
        serial.predict(_table([4.0])).numerical,
    )
    runner.close()


def test_cached_related_execution_forwards_two_hops() -> None:
    models = [_WorkerModel(0), _WorkerModel(1)]
    runner = EnsembleParallel(models, rng_policy="member")
    x_context = _id_table([0.0, 1.0], [0, 1])
    y_context = _table([0.0, 1.0])
    x_query = _id_table([2.0], [2])

    runner.fit(
        x_context,
        y_context,
        _related([10.0, 11.0], [0, 1]),
        num_estimators=4,
        generator=torch.Generator().manual_seed(0),
        num_hops=2,
    )
    output = runner.predict(x_query, _related([20.0], [2]))

    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[[22.0]], [[122.0]], [[22.0]], [[122.0]]]),
    )
    assert models[0].seen_num_hops == [2, 2, 2, 2]
    assert models[1].seen_num_hops == [2, 2, 2, 2]
    runner.close()


def test_rejects_gradient_callback() -> None:
    runner = EnsembleParallel([_WorkerModel(0)])

    with pytest.raises(ValueError, match="requiring gradients"):
        runner(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            _table([2.0]),
            callbacks=[_GradientCallback()],
        )

    runner.close()


def test_worker_errors_include_placement_and_fit_is_atomic() -> None:
    runner = EnsembleParallel(
        [_FailModel(0), _FailModel(1, fail_fit=True)],
        rng_policy="member",
    )

    with pytest.raises(
        RuntimeError,
        match=r"member 1 failed during fit on worker 1 \(cpu\)",
    ) as error:
        runner.fit(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            num_estimators=2,
            generator=torch.Generator().manual_seed(0),
        )
    assert isinstance(error.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match="not yet fitted"):
        runner.predict(_table([2.0]))
    runner.close()


def test_predict_error_and_serialization_are_explicit() -> None:
    models = [_FailModel(0), _FailModel(1)]
    runner = EnsembleParallel(models, rng_policy="member")
    runner.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        num_estimators=2,
        generator=torch.Generator().manual_seed(0),
    )
    models[1].fail_predict = True

    with pytest.raises(
        RuntimeError,
        match=r"member 1 failed during predict on worker 1 \(cpu\)",
    ) as error:
        runner.predict(_table([2.0]))
    assert isinstance(error.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match="cannot be serialized"):
        pickle.dumps(runner)

    runner.close()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("reduce", [False, True])
def test_canonical_rng_matches_icl_model(
    cached: bool,
    reduce: bool,
) -> None:
    x_context = _table([0.0, 1.0])
    y_context = _table([0.0, 1.0])
    x_query = _table([2.0, 3.0])

    serial = _RandomModel()
    serial_generator = torch.Generator().manual_seed(7)
    _GeneratorProcessor.reset()
    if cached:
        serial.fit(
            x_context,
            y_context,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=3,
            generator=serial_generator,
        )
        serial_output = serial.predict(x_query)
    else:
        serial_output = serial(
            x_context,
            y_context,
            x_query,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=3,
            generator=serial_generator,
        )
    serial_state = serial_generator.get_state().clone()
    serial_recipe_draws = torch.stack(_GeneratorProcessor.draws)

    runner = EnsembleParallel([_RandomModel()], rng_policy="canonical")
    runner_generator = torch.Generator().manual_seed(7)
    _GeneratorProcessor.reset()
    if cached:
        runner.fit(
            x_context,
            y_context,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=3,
            generator=runner_generator,
        )
        runner_output = runner.predict(x_query)
        assert len(runner.member_plan_digest) == 64
    else:
        runner_output = runner(
            x_context,
            y_context,
            x_query,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=3,
            generator=runner_generator,
        )
    runner_recipe_draws = torch.stack(_GeneratorProcessor.draws)

    torch.testing.assert_close(
        runner_output.numerical,
        serial_output.numerical,
    )
    torch.testing.assert_close(runner_recipe_draws, serial_recipe_draws)
    assert torch.equal(runner_generator.get_state(), serial_state)
    runner.close()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("reduce", [False, True])
def test_member_rng_is_worker_count_invariant(
    cached: bool,
    reduce: bool,
) -> None:
    x_context = _table([0.0, 1.0])
    y_context = _table([0.0, 1.0])
    x_query = _table([2.0, 3.0])
    one = EnsembleParallel([_RandomModel()], rng_policy="member")
    two = EnsembleParallel(
        [_RandomModel(), _RandomModel()],
        rng_policy="member",
    )

    one_generator = torch.Generator().manual_seed(11)
    _GeneratorProcessor.reset()
    if cached:
        one.fit(
            x_context,
            y_context,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=4,
            generator=one_generator,
        )
        one_output = one.predict(x_query)
    else:
        one_output = one(
            x_context,
            y_context,
            x_query,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=4,
            generator=one_generator,
        )
    one_recipe_draws = torch.stack(_GeneratorProcessor.draws)

    two_generator = torch.Generator().manual_seed(11)
    _GeneratorProcessor.reset()
    if cached:
        two.fit(
            x_context,
            y_context,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=4,
            generator=two_generator,
        )
        two_output = two.predict(x_query)
        assert one.member_plan == two.member_plan
        assert one.member_plan_digest == two.member_plan_digest
    else:
        two_output = two(
            x_context,
            y_context,
            x_query,
            recipe=_random_recipe(reduce=reduce),
            num_estimators=4,
            generator=two_generator,
        )
    two_recipe_draws = torch.stack(_GeneratorProcessor.draws)

    torch.testing.assert_close(one_output.numerical, two_output.numerical)
    torch.testing.assert_close(one_recipe_draws, two_recipe_draws)
    assert torch.equal(one_generator.get_state(), two_generator.get_state())
    one.close()
    two.close()


def test_rng_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires exactly one"):
        EnsembleParallel([_RandomModel(), _RandomModel()])

    runner = EnsembleParallel([_RandomModel()], rng_policy="member")
    with pytest.raises(ValueError, match="requires an explicit generator"):
        runner(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            _table([2.0]),
            num_estimators=2,
        )
    with pytest.raises(RuntimeError, match="requires a successful call"):
        _ = runner.member_plan_digest
    runner.close()


def test_member_plan_exposes_actual_seeds_and_reconstructs_digest() -> None:
    runner = EnsembleParallel([_RandomModel()], rng_policy="member")
    runner.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        recipe=_random_recipe(),
        num_estimators=3,
        generator=torch.Generator().manual_seed(17),
    )
    output = runner.predict(_table([2.0]))
    plan = runner.member_plan

    assert plan.policy == "member"
    assert len(plan.member_seeds) == 3
    assert all(isinstance(seed, int) for seed in plan.member_seeds)
    encoded = json.dumps(
        plan.digest_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == plan.digest_sha256
    assert plan.as_dict()["digest_sha256"] == plan.digest_sha256

    expected = []
    for seed in plan.member_seeds:
        assert seed is not None
        draw = torch.rand((2,), generator=torch.Generator().manual_seed(seed))
        expected.append(2.0 + draw.sum())
    torch.testing.assert_close(
        output.numerical,
        torch.stack(expected).view(3, 1, 1),
    )

    record = plan.as_dict()
    cast(list[int], record["member_seeds"])[0] = -1
    assert plan.member_seeds[0] != -1
    with pytest.raises(AttributeError):
        _assign_attribute(plan, "policy", "canonical")

    different = EnsembleParallel([_RandomModel()], rng_policy="member")
    different.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        recipe=_random_recipe(),
        num_estimators=3,
        generator=torch.Generator().manual_seed(18),
    )
    assert different.member_plan.member_seeds != plan.member_seeds
    assert different.member_plan_digest != plan.digest_sha256
    runner.close()
    different.close()


def test_rng_policy_is_read_only_and_revalidated_before_side_effects() -> None:
    models = [_RandomModel(), _RandomModel()]
    runner = EnsembleParallel(models, rng_policy="member")
    runner.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        num_estimators=2,
        generator=torch.Generator().manual_seed(3),
    )
    fitted_plan = runner.member_plan
    with pytest.raises(AttributeError):
        _assign_attribute(runner, "rng_policy", "canonical")

    callback = _CountingCallback()
    generator = torch.Generator().manual_seed(5)
    generator_state = generator.get_state().clone()
    calls = sum(model.calls for model in models)
    _GeneratorProcessor.reset()
    runner._rng_policy = cast(Any, "canonical")

    with pytest.raises(RuntimeError, match="requires exactly one"):
        runner(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            _table([2.0]),
            recipe=_random_recipe(),
            num_estimators=2,
            generator=generator,
            callbacks=[callback],
        )
    with pytest.raises(RuntimeError, match="requires exactly one"):
        runner.fit(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            recipe=_random_recipe(),
            num_estimators=2,
            generator=generator,
        )
    with pytest.raises(RuntimeError, match="requires exactly one"):
        runner.predict(_table([2.0]), callbacks=[callback])

    assert callback.starts == 0
    assert _GeneratorProcessor.draws == []
    assert torch.equal(generator.get_state(), generator_state)
    assert sum(model.calls for model in models) == calls
    assert runner.member_plan == fitted_plan

    runner._rng_policy = cast(Any, "bogus")
    with pytest.raises(AssertionError, match="Unsupported RNG policy"):
        runner(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            _table([2.0]),
            recipe=_random_recipe(),
            num_estimators=2,
            generator=generator,
            callbacks=[callback],
        )
    assert callback.starts == 0
    assert _GeneratorProcessor.draws == []
    assert torch.equal(generator.get_state(), generator_state)
    runner.close()


def test_predict_rejects_policy_different_from_fitted_plan() -> None:
    runner = EnsembleParallel([_RandomModel()], rng_policy="member")
    runner.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        num_estimators=2,
        generator=torch.Generator().manual_seed(0),
    )
    runner._rng_policy = cast(Any, "canonical")

    with pytest.raises(RuntimeError, match="does not match the fitted"):
        runner.predict(_table([2.0]))

    runner.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "module_list",
        "item",
        "reorder",
        "type",
        "count",
        "train",
        "dtype",
        "device",
        "load_state_dict",
        "parameter_in_place",
        "buffer_in_place",
        "storage",
    ],
)
def test_replica_mutations_fail_before_execution_side_effects(
    mutation: str,
) -> None:
    models = [_RandomModel(), _RandomModel()]
    runner = EnsembleParallel(models, rng_policy="member")
    runner.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        num_estimators=2,
        generator=torch.Generator().manual_seed(3),
    )
    fitted_state = runner._state
    fitted_plan = runner.member_plan

    if mutation == "module_list":
        runner.models = torch.nn.ModuleList([_RandomModel(), _RandomModel()])
    elif mutation == "item":
        runner.models[0] = _RandomModel()
    elif mutation == "reorder":
        runner.models[0], runner.models[1] = (
            runner.models[1],
            runner.models[0],
        )
    elif mutation == "type":
        runner.models[0] = _ReplacementRandomModel()
    elif mutation == "count":
        runner.models.append(_RandomModel())
    elif mutation == "train":
        models[0].train()
    elif mutation == "dtype":
        models[0].to(dtype=torch.float64)
    elif mutation == "device":
        models[0].to("meta")
    elif mutation == "load_state_dict":
        models[0].load_state_dict(models[0].state_dict())
    elif mutation == "parameter_in_place":
        with torch.no_grad():
            models[0].scale.add_(1)
    elif mutation == "buffer_in_place":
        cast(torch.Tensor, models[0].marker).add_(1)
    elif mutation == "storage":
        models[0].scale.data = models[0].scale.data.clone()
    else:
        raise AssertionError(f"Unhandled mutation {mutation!r}")

    callback = _CountingCallback()
    generator = torch.Generator().manual_seed(5)
    generator_state = generator.get_state().clone()
    observed_models = {
        id(model): model for model in (*models, *tuple(runner.models))
    }
    calls = {key: model.calls for key, model in observed_models.items()}
    _GeneratorProcessor.reset()

    with pytest.raises(RuntimeError, match="construct a new runner"):
        runner(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            _table([2.0]),
            recipe=_random_recipe(),
            num_estimators=2,
            generator=generator,
            callbacks=[callback],
        )
    with pytest.raises(RuntimeError, match="construct a new runner"):
        _ = runner.member_plan

    assert callback.starts == 0
    assert _GeneratorProcessor.draws == []
    assert torch.equal(generator.get_state(), generator_state)
    assert {
        key: model.calls for key, model in observed_models.items()
    } == calls
    assert runner._state is fitted_state
    assert fitted_state is not None
    assert fitted_state.member_plan == fitted_plan
    runner.close()


@pytest.mark.parametrize("entry", ["forward", "fit", "predict"])
def test_replica_authority_is_checked_at_every_entry(entry: str) -> None:
    model = _RandomModel()
    runner = EnsembleParallel([model], rng_policy="member")
    runner.fit(
        _table([0.0, 1.0]),
        _table([0.0, 1.0]),
        num_estimators=2,
        generator=torch.Generator().manual_seed(3),
    )
    fitted_state = runner._state
    cast(torch.Tensor, model.marker).add_(1)
    callback = _CountingCallback()
    generator = torch.Generator().manual_seed(5)
    generator_state = generator.get_state().clone()
    calls = model.calls
    _GeneratorProcessor.reset()

    def execute() -> None:
        if entry == "forward":
            runner(
                _table([0.0, 1.0]),
                _table([0.0, 1.0]),
                _table([2.0]),
                recipe=_random_recipe(),
                num_estimators=2,
                generator=generator,
                callbacks=[callback],
            )
        elif entry == "fit":
            runner.fit(
                _table([0.0, 1.0]),
                _table([0.0, 1.0]),
                recipe=_random_recipe(),
                num_estimators=2,
                generator=generator,
            )
        else:
            runner.predict(_table([2.0]), callbacks=[callback])

    with pytest.raises(RuntimeError, match="construct a new runner"):
        execute()

    assert callback.starts == 0
    assert _GeneratorProcessor.draws == []
    assert torch.equal(generator.get_state(), generator_state)
    assert model.calls == calls
    assert runner._state is fitted_state
    runner.close()


def test_fit_rechecks_model_authority_before_publishing_state() -> None:
    model = _FitMutatingRandomModel()
    runner = EnsembleParallel([model], rng_policy="member")

    with pytest.raises(RuntimeError, match="construct a new runner"):
        runner.fit(
            _table([0.0, 1.0]),
            _table([0.0, 1.0]),
            num_estimators=2,
            generator=torch.Generator().manual_seed(3),
        )

    assert runner._state is None
    with pytest.raises(RuntimeError, match="construct a new runner"):
        _ = runner.member_plan_digest
    runner.close()
