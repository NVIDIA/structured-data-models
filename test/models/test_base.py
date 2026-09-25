# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import pytest
import torch

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    EnsembleTable,
    RelatedTables,
    Stype,
    TableTensor,
)
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models.callback import Callback
from sdm.processing import InvertibleMixin, Processor


@dataclass
class _Call:
    x_context: TableTensor | None
    x_query: TableTensor | None
    related_context_tables: RelatedTables | None
    related_query_tables: RelatedTables | None


class _RecordingModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_multi_target = False
    supports_related_tables = True

    def __init__(self) -> None:
        super().__init__(task=None)
        self.calls: list[_Call] = []
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


class _ClassFrequencyModel(ICLModel):
    """Predict the context class frequencies, one column per class."""

    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.categorical})
    supports_multi_target = False
    supports_related_tables = False

    def __init__(self) -> None:
        super().__init__(task=None)
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
        if cache is None or cache.is_recording:
            assert y_context is not None
            classes = y_context.categorical.categories[0]
            code = y_context.categorical.code.squeeze(-1).long()  # [..., R]
            counts = torch.nn.functional.one_hot(code, len(classes)).float()
            frequency = counts.mean(dim=-2, keepdim=True)  # [..., 1, K]
            if cache is not None:
                cache["frequency"] = frequency
        else:
            classes = cast(torch.Tensor, cache["classes"])
            frequency = cast(torch.Tensor, cache["frequency"])
        if x_query is None:
            return TableTensor(numerical=frequency)
        return TableTensor(
            columns={Stype.numerical: [str(c) for c in classes.tolist()]},
            numerical=frequency.expand(*x_query.size()[:-1], -1),
        )

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


@pytest.mark.parametrize("num_estimators", [1, 3])
@pytest.mark.parametrize("estimator_batch_size", [1, 2, None])
def test_train_mode_enables_grad(
    num_estimators: int, estimator_batch_size: int | None
) -> None:
    model = _RecordingModel()
    x_context = torch.tensor([[0.0], [2.0]])
    y_context = torch.tensor([[0.0], [1.0]])
    x_query = torch.tensor([[3.0]])

    model.eval()
    out = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        num_estimators=num_estimators,
        estimator_batch_size=estimator_batch_size,
    )
    assert torch.is_inference(out)

    model.train()
    out = model(
        x_context=x_context,
        y_context=y_context,
        x_query=x_query,
        num_estimators=num_estimators,
        estimator_batch_size=estimator_batch_size,
    )
    assert not torch.is_inference(out)


def test_train_mode_disallowed_for_predict() -> None:
    model = _RecordingModel()
    x_context = torch.tensor([[0.0], [2.0]])
    y_context = torch.tensor([[0.0], [1.0]])
    x_query = torch.tensor([[3.0]])

    model.eval()
    model.fit(x_context, y_context)
    out = model.predict(x_query)
    assert torch.is_inference(out)

    model.train()
    with pytest.raises(RuntimeError, match="does not support"):
        model.predict(x_query)


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


def test_ensemble_output_reduce() -> None:
    x_context = torch.randn(4, 3)
    y_context = torch.randn(4, 1)
    x_query = torch.randn(2, 3)

    model = _RecordingModel()
    out = model(
        x_context,
        y_context,
        x_query,
        recipe=sp.Recipe(output=sp.AverageEstimators()),
        num_estimators=2,
    )

    assert out.size() == (2, 3)


@pytest.mark.parametrize("estimator_batch_size", [1, 2, None])
def test_estimator_callbacks(estimator_batch_size: int | None) -> None:
    model = _RecordingModel()
    x = torch.arange(30.0).view(5, 3, 2)
    y = torch.zeros(5, 3, 1)
    events: list[str] = []
    callbacks = (MyCallback("affine", 2.0, 3.0, events),)
    hooks = (
        "context_preprocessing_end",
        "query_preprocessing_end",
        "model_forward_end",
    )

    out = model(
        x_context=x,
        y_context=y,
        x_query=x,
        estimator_batch_size=estimator_batch_size,
        callbacks=callbacks,
    )
    torch.testing.assert_close(out.numerical, 2.0 * x + 3.0)
    for hook in hooks:
        assert events.count(f"affine_{hook}") == 5

    events.clear()
    model.fit(
        x=x,
        y=y,
        estimator_batch_size=estimator_batch_size,
        callbacks=callbacks,
    )
    out = model.predict(x, callbacks=callbacks)
    torch.testing.assert_close(out.numerical, 2.0 * x + 3.0)
    for hook in hooks:
        assert events.count(f"affine_{hook}") == 5


@pytest.mark.parametrize(
    ("estimator_batch_size", "num_calls", "query_size"),
    [(1, 4, (2, 2)), (2, 2, (2, 2, 2)), (None, 1, (4, 2, 2))],
)
def test_estimator_batching_groups_consecutive_members(
    estimator_batch_size: int | None,
    num_calls: int,
    query_size: tuple[int, ...],
) -> None:
    model = _RecordingModel()
    x = torch.randn(4, 3, 2)
    y = torch.zeros(4, 3, 1)
    x_query = torch.randn(4, 2, 2)

    out = model(x, y, x_query, estimator_batch_size=estimator_batch_size)

    torch.testing.assert_close(out.numerical, x_query)
    assert len(model.calls) == num_calls
    assert model.calls[0].x_query is not None
    assert model.calls[0].x_query.size() == query_size

    model.calls.clear()
    model.fit(x, y, estimator_batch_size=estimator_batch_size)
    out = model.predict(x_query)

    torch.testing.assert_close(out.numerical, x_query)
    assert len(model.calls) == 2 * num_calls
    assert model.calls[-1].x_query is not None
    assert model.calls[-1].x_query.size() == query_size


@pytest.mark.parametrize("estimator_batch_size", [2, None])
def test_estimator_batching_relabels_shuffled_classes(
    estimator_batch_size: int | None,
) -> None:
    model = _ClassFrequencyModel()
    x = torch.randn(4, 2)
    y = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0], [0]]),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    # Shift the class order of every other estimator.
    recipe = sp.Recipe(
        target=sp.StypeDispatch(
            categorical=sp.ShuffleCategories(method="shift")
        ),
    )

    def _check(out: TableTensor) -> None:
        columns = out.columns[Stype.numerical]
        assert sorted(columns) == ["10", "20"]
        frequency = torch.tensor(
            [0.75 if column == "10" else 0.25 for column in columns]
        )
        torch.testing.assert_close(out.numerical, frequency.expand(4, 4, -1))

    _check(
        model(
            x_context=x,
            y_context=y,
            x_query=x,
            recipe=recipe,
            num_estimators=4,
            estimator_batch_size=estimator_batch_size,
            generator=torch.Generator().manual_seed(0),
        )
    )

    model.fit(
        x=x,
        y=y,
        recipe=recipe,
        num_estimators=4,
        estimator_batch_size=estimator_batch_size,
        generator=torch.Generator().manual_seed(0),
    )
    _check(model.predict(x))


def test_estimator_batching_rejects_related_tables() -> None:
    model = _RecordingModel()
    x_context = _table([0.0, 2.0], [1, 2], value_column="feature")
    x_query = _table([3.0], [3], value_column="feature")
    y_context = TableTensor.from_tensor(torch.tensor([[0.0], [1.0]]))
    related_context = _related_tables(query=False)
    related_query = _related_tables(query=True)

    with pytest.raises(ValueError, match="Related tables require"):
        model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context,
            related_query_tables=related_query,
            num_estimators=2,
            estimator_batch_size=None,
        )
    with pytest.raises(ValueError, match="Related tables require"):
        model.fit(
            x=x_context,
            y=y_context,
            related_tables=related_context,
            num_estimators=2,
            estimator_batch_size=None,
        )


def test_estimator_batching_incompatible_shapes() -> None:
    model = _RecordingModel()
    x = EnsembleTable.from_tables(
        tables=[
            TableTensor(numerical=torch.ones(3, 2)),
            TableTensor(numerical=torch.ones(4, 2)),
        ],
        member_table_ids=(0, 1),
    )
    y = EnsembleTable.from_tables(
        tables=[
            TableTensor(numerical=torch.ones(3, 1)),
            TableTensor(numerical=torch.ones(4, 1)),
        ],
        member_table_ids=(0, 1),
    )
    with pytest.raises(RuntimeError, match="stack expects"):
        model.fit(x, y, estimator_batch_size=None)
    query = TableTensor(numerical=torch.randn(2, 2, 2))
    with pytest.raises(RuntimeError, match="stack expects"):
        model(x, y, query, estimator_batch_size=None)
    model.fit(x, y)
    torch.testing.assert_close(model.predict(query).numerical, query.numerical)


def test_estimator_batching_incompatible_categories() -> None:
    model = _RecordingModel()
    y = EnsembleTable.from_tables(
        tables=[
            TableTensor(
                categorical=CategoricalTensor(
                    code=torch.zeros(3, 1, dtype=torch.long),
                    categories=(torch.arange(count),),
                ),
            )
            for count in (2, 3)
        ],
        member_table_ids=(0, 1),
    )
    with pytest.raises(ValueError, match="Estimators in one batch"):
        model.fit(
            x=torch.ones(3, 2),
            y=y,
            num_estimators=2,
            estimator_batch_size=None,
        )
    with pytest.raises(ValueError, match="Estimators in one batch"):
        model(
            x_context=torch.ones(3, 2),
            y_context=y,
            x_query=torch.ones(1, 2),
            num_estimators=2,
            estimator_batch_size=None,
        )


@pytest.mark.parametrize("columns", [("a", "c"), ("a",)])
def test_estimator_batching_incompatible_columns(
    columns: tuple[str, ...],
) -> None:
    model = _RecordingModel()
    tables = [
        TableTensor(
            columns={Stype.numerical: ("a", "b")},
            numerical=torch.ones(3, 2),
        ),
        TableTensor(
            columns={Stype.numerical: columns},
            numerical=torch.ones(3, len(columns)),
        ),
    ]
    x = EnsembleTable.from_tables(tables=tables, member_table_ids=(0, 1))
    y = torch.zeros(2, 3, 1)
    x_query = EnsembleTable.from_tables(
        tables=[table[:1] for table in tables],
        member_table_ids=(0, 1),
    )
    with pytest.raises(ValueError, match="Estimators in one batch"):
        model(x, y, x_query, estimator_batch_size=None)
    with pytest.raises(ValueError, match="Estimators in one batch"):
        model.fit(x, y, estimator_batch_size=None)


def test_estimator_batching_incompatible_class_values() -> None:
    model = _ClassFrequencyModel()
    x = torch.randn(3, 2)
    y = EnsembleTable.from_tables(
        tables=[
            TableTensor(
                categorical=CategoricalTensor(
                    code=torch.tensor([[0], [1], [0]]),
                    categories=(torch.tensor(categories),),
                ),
            )
            for categories in ([10, 20], [10, 30])
        ],
        member_table_ids=(0, 1),
    )
    with pytest.raises(ValueError, match="Estimators in one batch"):
        model(x, y, x, num_estimators=2, estimator_batch_size=None)
    with pytest.raises(ValueError, match="Estimators in one batch"):
        model.fit(x, y, num_estimators=2, estimator_batch_size=None)


@pytest.mark.parametrize("estimator_batch_size", [1, 2, None])
def test_forward_batching_validates_each_query_schema(
    estimator_batch_size: int | None,
) -> None:
    model = _RecordingModel()
    context = TableTensor(
        columns={Stype.numerical: ("a", "b")},
        numerical=torch.ones(3, 2),
    )
    x = EnsembleTable.from_tables(
        tables=[
            context,
            TableTensor(
                columns={Stype.numerical: ("b", "a")},
                numerical=torch.ones(3, 2),
            ),
        ],
        member_table_ids=(0, 1),
    )
    y = torch.zeros(2, 3, 1)
    query = EnsembleTable.from_tables(
        tables=[context[:1]],
        member_table_ids=(0, 0),
    )
    with pytest.raises(ValueError, match="share the same schema"):
        model(x, y, query, estimator_batch_size=estimator_batch_size)
