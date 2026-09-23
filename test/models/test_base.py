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
def test_train_mode_enables_grad(num_estimators: int) -> None:
    model = _RecordingModel()
    x_context = torch.tensor([[0.0], [2.0]])
    y_context = torch.tensor([[0.0], [1.0]])
    x_query = torch.tensor([[3.0]])

    model.eval()
    out = model(x_context, y_context, x_query, num_estimators=num_estimators)
    assert torch.is_inference(out)

    model.train()
    out = model(x_context, y_context, x_query, num_estimators=num_estimators)
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
    callbacks = (MyCallback("affine", 2.0, 3.0, []),)
    if estimator_batch_size != 1:
        with pytest.raises(ValueError, match="Callbacks require"):
            model.fit(
                x=x,
                y=y,
                estimator_batch_size=estimator_batch_size,
                callbacks=callbacks,
            )
    model.fit(x, y, estimator_batch_size=estimator_batch_size)
    if estimator_batch_size != 1:
        with pytest.raises(ValueError, match="Callbacks require"):
            model.predict(
                x=x,
                estimator_batch_size=estimator_batch_size,
                callbacks=callbacks,
            )
    out = model.predict(x, callbacks=callbacks)
    torch.testing.assert_close(out.numerical, 2.0 * x + 3.0)
    out = model(
        x_context=x,
        y_context=y,
        x_query=x,
        callbacks=callbacks,
    )
    torch.testing.assert_close(out.numerical, 2.0 * x + 3.0)


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
    model.fit(x, y)
    query = torch.randn(2, 2, 2)
    torch.testing.assert_close(
        model.predict(TableTensor(numerical=query)).numerical,
        query,
    )


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
    with pytest.raises(ValueError, match="matching class counts"):
        model.fit(
            torch.ones(3, 2), y, num_estimators=2, estimator_batch_size=None
        )
    model.fit(torch.ones(3, 2), y, num_estimators=2)
    with pytest.raises(ValueError, match="compatible caches"):
        model.predict(torch.ones(1, 2), estimator_batch_size=None)
