from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import pytest
import torch

import sdm.processing as sp
from sdm import (
    ColumnarTensor,
    EnsembleTable,
    RelatedTables,
    Stype,
    TableTensor,
)
from sdm.cache import Cache
from sdm.callbacks import Callback
from sdm.models import ICLModel
from sdm.processing import InvertibleMixin, Processor


@dataclass
class _Call:
    x_context: TableTensor | None
    x_query: TableTensor | None
    related_context_tables: RelatedTables | None
    related_query_tables: RelatedTables | None
    kwargs: dict[str, Any] = field(default_factory=dict)


class _RecordingModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_multi_target = False
    supports_related_tables = True

    def __init__(self) -> None:
        super().__init__(task=None)
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
                kwargs=dict(kwargs),
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
    supports_multi_target = False
    supports_related_tables = False


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

    def _record(self, event: str) -> None:
        self.events.append(f"{self.name}_{event}")

    def on_context_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, TableTensor, RelatedTables | None]:
        self._record("context_preprocessing_end")
        return x, y, related_tables

    def on_query_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        self._record("query_preprocessing_end")
        return (
            x.replace_blocks(numerical=x.numerical * self.scale + self.offset),
            related_tables,
        )

    def on_model_forward_end(
        self,
        model: torch.nn.Module,
        out: TableTensor,
    ) -> TableTensor:
        self._record("model_forward_end")
        return out


class _LegacyModel(ICLModel):
    """Model implementing the private hook without keyword passthrough."""

    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_multi_target: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = False

    # Deliberately narrower than `ICLModel._forward`: this model predates the
    # optional `seqused_train`/`seqused_cols` (and any other model-specific)
    # keywords and pins down that they are only forwarded when the caller
    # asks for them.
    def _forward(  # ty: ignore[invalid-method-override]
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
    ) -> TableTensor:
        del y_context, related_context_tables, related_query_tables
        del cache, generator
        table = x_query if x_query is not None else x_context
        assert table is not None
        return table.select_stypes(Stype.numerical)

    @classmethod
    def default_recipe(cls) -> sp.Recipe:
        return sp.Recipe()


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

    output = model(
        x_context,
        y_context,
        x_query,
        callbacks=callbacks,
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

    torch.testing.assert_close(output.numerical, torch.tensor([[[8.0]]]))
    torch.testing.assert_close(prediction.numerical, output.numerical)

    assert events == 2 * [
        "1_context_preprocessing_end",
        "2_context_preprocessing_end",
        "1_query_preprocessing_end",
        "2_query_preprocessing_end",
        "1_model_forward_end",
        "2_model_forward_end",
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


def test_legacy_model_forward_compatibility() -> None:
    """The default path does not pass new keywords to existing subclasses."""
    model = _LegacyModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)

    out = model(x_context, y_context, x_query)
    torch.testing.assert_close(out.numerical, x_query.unsqueeze(0))

    model.fit(x_context, y_context)
    prediction = model.predict(x_query)
    torch.testing.assert_close(prediction.numerical, x_query.unsqueeze(0))

    # Opting into padding (or other model-specific) keywords surfaces a hard
    # error on subclasses that do not implement them instead of silently
    # dropping them.
    with pytest.raises(TypeError, match="unsupported_kwarg"):
        model(x_context, y_context, x_query, unsupported_kwarg=2)
    with pytest.raises(TypeError, match="unsupported_kwarg"):
        model.fit(x_context, y_context, unsupported_kwarg=2)


def test_forward_forwards_padding_keywords() -> None:
    """Padding keywords reach the model only when explicitly requested."""
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)
    seqused_train = torch.tensor(3, dtype=torch.int32)
    seqused_cols = torch.tensor(2, dtype=torch.int32)

    model(x_context, y_context, x_query)

    assert len(model.calls) == 1
    assert model.calls[0].kwargs == {}

    model.calls.clear()
    model(
        x_context,
        y_context,
        x_query,
        seqused_train=seqused_train,
        seqused_cols=seqused_cols,
        model_specific_kwarg=2,
    )

    assert len(model.calls) == 1
    kwargs = model.calls[0].kwargs
    assert kwargs["model_specific_kwarg"] == 2
    assert torch.equal(kwargs["seqused_train"], seqused_train)
    assert torch.equal(kwargs["seqused_cols"], seqused_cols)
    assert kwargs["seqused_train"].dtype == torch.int32
    assert kwargs["seqused_cols"].dtype == torch.int32


def test_fit_caches_padding_keywords_for_predict() -> None:
    """Padding counts given to ``fit`` are replayed by ``predict``."""
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)
    seqused_train = torch.tensor(3, dtype=torch.int32)
    seqused_cols = torch.tensor(2, dtype=torch.int32)

    model.fit(
        x_context,
        y_context,
        seqused_train=seqused_train,
        seqused_cols=seqused_cols,
    )

    assert len(model.calls) == 1
    fit_kwargs = model.calls[0].kwargs
    assert torch.equal(fit_kwargs["seqused_train"], seqused_train)
    assert torch.equal(fit_kwargs["seqused_cols"], seqused_cols)

    # The cache holds its own copy of the counts: mutating the caller's
    # tensors between `fit` and `predict` must not change replay masking.
    model.calls.clear()
    seqused_train.fill_(1)
    seqused_cols.fill_(1)
    model.predict(x_query)

    assert len(model.calls) == 1
    predict_kwargs = model.calls[0].kwargs
    assert predict_kwargs["seqused_train"].item() == 3
    assert predict_kwargs["seqused_cols"].item() == 2


def test_seqused_rejects_ensemble_aware_inputs() -> None:
    """Padding counts describe one shared context, not per-member tables."""
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)
    seqused_train = torch.tensor(3, dtype=torch.int32)
    ensemble = EnsembleTable.from_table(
        TableTensor.from_tensor(x_context),
        num_members=2,
    )

    with pytest.raises(ValueError, match="EnsembleTable"):
        model(ensemble, y_context, x_query, seqused_train=seqused_train)
    with pytest.raises(ValueError, match="EnsembleTable"):
        model.fit(ensemble, y_context, seqused_train=seqused_train)
    assert len(model.calls) == 0

    # Higher-rank inputs are only unambiguous with explicit
    # 'num_estimators'; otherwise the leading dimension would be consumed
    # as the estimator dimension while the counts index batch elements.
    with pytest.raises(ValueError, match="num_estimators"):
        model(
            torch.randn(2, 4, 3),
            torch.randn(2, 4, 1),
            torch.randn(2, 2, 3),
            seqused_train=torch.tensor([3, 4], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="num_estimators"):
        model.fit(
            torch.randn(2, 4, 3),
            torch.randn(2, 4, 1),
            seqused_train=torch.tensor([3, 4], dtype=torch.int32),
        )
    assert len(model.calls) == 0

    model(
        torch.randn(2, 4, 3),
        torch.randn(2, 4, 1),
        torch.randn(2, 2, 3),
        num_estimators=1,
        seqused_train=torch.tensor([3, 4], dtype=torch.int32),
    )
    assert len(model.calls) == 1


def test_seqused_train_validates_batch_shape() -> None:
    """The per-element counts must match the batch dimensions."""
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)

    # Unbatched tables have no batch dimensions, so the count must be a
    # scalar.
    with pytest.raises(ValueError, match="batch dimensions"):
        model(
            x_context,
            y_context,
            x_query,
            seqused_train=torch.tensor([3, 4], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="batch dimensions"):
        model.fit(
            x_context,
            y_context,
            seqused_train=torch.tensor([3, 4], dtype=torch.int32),
        )
    # `TableTensor` inputs carry the same trailing row and column
    # dimensions, so the check applies to them as well.
    with pytest.raises(ValueError, match="batch dimensions"):
        model(
            TableTensor.from_tensor(x_context),
            y_context,
            x_query,
            seqused_train=torch.tensor([3, 4], dtype=torch.int32),
        )
    # Batched tables require one count per batch element.
    with pytest.raises(ValueError, match="batch dimensions"):
        model(
            torch.randn(2, 4, 3),
            torch.randn(2, 4, 1),
            torch.randn(2, 2, 3),
            num_estimators=1,
            seqused_train=torch.tensor([3, 4, 5], dtype=torch.int32),
        )
    assert len(model.calls) == 0

    model(
        torch.randn(2, 4, 3),
        torch.randn(2, 4, 1),
        torch.randn(2, 2, 3),
        num_estimators=1,
        seqused_train=torch.tensor([3, 4], dtype=torch.int32),
    )
    assert len(model.calls) == 1


class _RowReversingCallback(Callback):
    """Returns a context with reversed rows, keeping shape and schema."""

    def on_context_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        y: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, TableTensor, RelatedTables | None]:
        return (
            x.replace_blocks(numerical=x.numerical.flip(-2)),
            y.replace_blocks(numerical=y.numerical.flip(-2)),
            related_tables,
        )


def test_seqused_rejects_context_replacing_callbacks() -> None:
    """A row-reversing callback would move padding into the valid prefix."""
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)
    seqused_train = torch.tensor(3, dtype=torch.int32)
    seqused_cols = torch.tensor(2, dtype=torch.int32)
    callback = _RowReversingCallback()

    # Without padding counts, replacement contexts remain legal.
    model(x_context, y_context, x_query, callbacks=(callback,))
    assert len(model.calls) == 1

    # With counts set, the replacement preserves shape and schema, so no
    # downstream validation could tell it apart from padding-safe output.
    # It must be rejected before the model call.
    model.calls.clear()
    with pytest.raises(ValueError, match="replacement context"):
        model(
            x_context,
            y_context,
            x_query,
            callbacks=(callback,),
            seqused_train=seqused_train,
        )
    with pytest.raises(ValueError, match="replacement context"):
        model(
            x_context,
            y_context,
            x_query,
            callbacks=(callback,),
            seqused_cols=seqused_cols,
        )
    with pytest.raises(ValueError, match="replacement context"):
        model.fit(
            x_context,
            y_context,
            callbacks=(callback,),
            seqused_train=seqused_train,
        )
    assert len(model.calls) == 0

    # Passive observers return the context unchanged and keep working; the
    # query hook stays open under row counts for layout-preserving
    # replacement (`MyCallback` replaces the query features).
    events: list[str] = []
    observer = MyCallback("obs", scale=1.0, offset=0.0, events=events)
    model(
        x_context,
        y_context,
        x_query,
        callbacks=(observer,),
        seqused_train=seqused_train,
    )
    assert len(model.calls) == 1
    assert "obs_context_preprocessing_end" in events


class _ColumnReversingQueryCallback(Callback):
    """Returns a query with reversed columns, keeping shape and schema."""

    def on_query_preprocessing_end(
        self,
        model: torch.nn.Module,
        x: TableTensor,
        related_tables: RelatedTables[TableTensor] | None,
    ) -> tuple[TableTensor, RelatedTables | None]:
        return (
            x.replace_blocks(numerical=x.numerical.flip(-1)),
            related_tables,
        )


def test_seqused_cols_rejects_query_replacing_callbacks() -> None:
    """A column-reversing query callback would move padding into the prefix.

    `seqused_cols` counts the valid columns of the query as well, so the
    guard covers the query hook, including the fit/predict replay path.
    """
    model = _RecordingModel()
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)
    seqused_cols = torch.tensor(2, dtype=torch.int32)
    callback = _ColumnReversingQueryCallback()

    # Without padding counts, replacement queries remain legal.
    model(x_context, y_context, x_query, callbacks=(callback,))
    assert len(model.calls) == 1

    # With the column count set, the replacement preserves shape and
    # schema, so it must be rejected before the model call.
    model.calls.clear()
    with pytest.raises(ValueError, match="replacement query"):
        model(
            x_context,
            y_context,
            x_query,
            callbacks=(callback,),
            seqused_cols=seqused_cols,
        )
    assert len(model.calls) == 0

    # `fit` caches the counts and `predict` replays them, even though the
    # `predict` caller never passed them: rejected there as well.
    model.fit(x_context, y_context, seqused_cols=seqused_cols)
    model.calls.clear()
    with pytest.raises(ValueError, match="replacement query"):
        model.predict(x_query, callbacks=(callback,))
    assert len(model.calls) == 0

    # Passive query observers return the inputs unchanged and keep working
    # with column counts.
    model.calls.clear()
    model(
        x_context,
        y_context,
        x_query,
        callbacks=(Callback(),),
        seqused_cols=seqused_cols,
    )
    assert len(model.calls) == 1

    # `seqused_train` counts context rows, which a replaced query cannot
    # invalidate: layout-preserving query replacement stays open on both
    # the forward and the fit/predict replay path.
    events: list[str] = []
    replacer = MyCallback("rep", scale=2.0, offset=1.0, events=events)
    seqused_train = torch.tensor(3, dtype=torch.int32)
    model.calls.clear()
    model(
        x_context,
        y_context,
        x_query,
        callbacks=(replacer,),
        seqused_train=seqused_train,
    )
    assert len(model.calls) == 1
    model.fit(x_context, y_context, seqused_train=seqused_train)
    model.calls.clear()
    model.predict(x_query, callbacks=(replacer,))
    assert len(model.calls) == 1
    assert "rep_query_preprocessing_end" in events
