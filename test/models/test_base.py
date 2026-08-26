from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import pytest
import torch

import sdm.processing as sp
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache, KVCacheEntry, KVCacheOffload
from sdm.callbacks import Callback
from sdm.models import ICLModel
from sdm.processing import InvertibleMixin, Processor
from sdm.testing import withCUDA


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
    def default_recipe(cls) -> sp.Recipe:
        return sp.Recipe()


class _UnsupportedRecordingModel(_RecordingModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables = False


class _KVRecordingModel(_RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_placements: list[tuple[torch.device, bool]] = []

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
        if cache is not None and cache.is_recording:
            value = out.numerical.unsqueeze(-2)
            cache["key_value"] = KVCacheEntry(key=value, value=value)
            entry = cast(KVCacheEntry, cache["key_value"])
            self.recorded_placements.append(
                (entry.key.device, entry.key.is_pinned())
            )
            cache["other"] = value
        return out


class MyCallback(Callback):
    """Callback used by callback lifecycle tests."""

    def __init__(
        self,
        name: str,
        scale: float,
        offset: float,
        events: list[str],
    ) -> None:
        self.name = name
        self.scale = scale
        self.offset = offset
        self.events = events
        self.start_calls: list[
            tuple[torch.nn.Module, tuple[Any, ...], dict[str, Any]]
        ] = []

    def _record(self, event: str) -> None:
        self.events.append(f"{self.name}_{event}")

    def on_forward_start(
        self,
        model: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.start_calls.append((model, args, kwargs))
        self._record("forward_start")

    def on_forward_end(
        self,
        model: torch.nn.Module,
        prediction: TableTensor,
    ) -> None:
        self._record("forward_end")

    def on_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        self._record("preprocessing_end")
        return (
            x.replace_blocks(numerical=x.numerical * self.scale + self.offset),
            related_tables,
        )


class _GeneratorRecordingProcessor(Processor, InvertibleMixin):
    handles_stypes = frozenset(Stype)
    requires_fit = True
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


def _recipe() -> sp.Recipe:
    return sp.Recipe(
        features=sp.StypeDispatch(numerical=sp.Standardize()),
    )


def _generator_recipe() -> sp.Recipe:
    return sp.Recipe(
        features=_GeneratorRecordingProcessor(),
        target=_GeneratorRecordingProcessor(),
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

    assert _GeneratorRecordingProcessor.generators == [generator] * 4
    return list(_GeneratorRecordingProcessor.draws)


@pytest.mark.parametrize("cached", [False, True])
def test_model_recipe_fitting_honors_generator(cached: bool) -> None:
    first = _fit_draws(seed=0, cached=cached)
    second = _fit_draws(seed=0, cached=cached)
    different_seed = _fit_draws(seed=1, cached=cached)

    assert len(first) == 4
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


def test_callback() -> None:
    events: list[str] = []
    callbacks = (
        MyCallback("1", scale=1.0, offset=1.0, events=events),
        MyCallback("2", scale=2.0, offset=0.0, events=events),
    )
    model = _RecordingModel()
    x_context = torch.tensor([[0.0], [2.0]])
    y_context = torch.tensor([[0.0], [1.0]])
    x_query = torch.tensor([[3.0]])
    marker = object()

    output = model(
        x_context,
        y_context,
        x_query,
        callbacks=callbacks,
        marker=marker,
    )
    model.fit(
        x_context,
        y_context,
        callbacks=callbacks,
    )
    prediction = model.predict(
        x_query,
        callbacks=callbacks,
    )
    with pytest.raises(RuntimeError, match="not yet fitted"):
        _RecordingModel().predict(
            torch.tensor([[3.0]]),
            callbacks=callbacks,
        )

    torch.testing.assert_close(output.numerical, torch.tensor([[[8.0]]]))
    torch.testing.assert_close(prediction.numerical, output.numerical)
    callback = callbacks[0]
    start_model, start_args, start_kwargs = callback.start_calls[0]
    assert start_model is model
    assert start_args[0] is x_context
    assert start_args[1] is y_context
    assert start_args[2] is x_query
    assert start_args[3:] == (None, None)
    assert start_kwargs == {
        "recipe": None,
        "num_estimators": 1,
        "generator": None,
        "callbacks": callbacks,
        "marker": marker,
    }

    predict_model, predict_args, predict_kwargs = callback.start_calls[1]
    assert predict_model is model
    assert predict_args[0] is x_query
    assert predict_args[1] is None
    assert predict_kwargs == {"callbacks": callbacks}
    assert events == [
        "1_forward_start",
        "2_forward_start",
        "1_preprocessing_end",
        "2_preprocessing_end",
        "1_forward_end",
        "2_forward_end",
        "1_forward_start",
        "2_forward_start",
        "1_preprocessing_end",
        "2_preprocessing_end",
        "1_forward_end",
        "2_forward_end",
        "1_forward_start",
        "2_forward_start",
    ]


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
    assert model._cache is not None

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


def test_task_dispatch() -> None:
    model = _RecordingModel()
    recipe = sp.Recipe(
        features=sp.TaskDispatch(regression=sp.Standardize()),
        output=sp.TaskDispatch(regression=sp.Identity()),
    )

    output = model(
        x_context=torch.tensor([[0.0], [2.0]]),
        y_context=torch.tensor([[0.0], [1.0]]),
        x_query=torch.tensor([[3.0]]),
        recipe=recipe,
    )

    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[[2.0]]]),
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


@withCUDA
@pytest.mark.parametrize(
    ("offload", "num_estimators"),
    [
        ("auto", 1),
        ("auto", 2),
        ("none", 2),
        ("estimator", 1),
        ("layer", 1),
    ],
)
def test_fit_kv_cache_offload(
    device: torch.device,
    offload: Literal["auto"] | KVCacheOffload,
    num_estimators: int,
) -> None:
    model = _KVRecordingModel()
    model.fit(
        torch.arange(32, device=device, dtype=torch.float32).view(4, 8),
        torch.arange(4, device=device, dtype=torch.float32).view(4, 1),
        num_estimators=num_estimators,
        kv_cache_offload=offload,
    )

    assert len(model.recorded_placements) == num_estimators
    if device.type == "cuda":
        for recorded_device, pinned in model.recorded_placements:
            assert (recorded_device.type == "cpu") == (offload == "layer")
            assert pinned == (offload == "layer")

    assert model._cache is not None
    estimator_cache = cast(Cache, model._cache[0])
    entry = cast(KVCacheEntry, estimator_cache["key_value"])
    other = cast(torch.Tensor, estimator_cache["other"])
    offloaded = device.type == "cuda" and (
        offload in ("estimator", "layer")
        or (offload == "auto" and num_estimators > 1)
    )
    if offloaded:
        assert entry.key.is_cpu
        assert entry.value.is_cpu
        assert other.is_cpu
        assert entry.key.is_pinned()
        assert entry.value.is_pinned()
        assert other.is_pinned()
    else:
        assert entry.key.device == device
        assert entry.value.device == device
        assert other.device == device

    prediction = model.predict(
        torch.arange(16, device=device, dtype=torch.float32).view(2, 8)
    )
    assert prediction.device == device


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


def test_ensemble_output_preserves_estimator_dimension() -> None:
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)

    model = _RecordingModel()
    out = model(
        x_context,
        y_context,
        x_query,
        recipe=sp.Recipe(),
        num_estimators=1,
    )

    assert out.size() == (1, 2, 3)


def test_ensemble_output_reduces_with_reduce_estimators() -> None:
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)

    model = _RecordingModel()
    out = model(
        x_context,
        y_context,
        x_query,
        recipe=sp.Recipe(output=sp.ReduceEstimators()),
        num_estimators=2,
    )

    assert out.size() == (2, 3)
