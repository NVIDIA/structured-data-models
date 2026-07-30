from typing import Any, ClassVar, cast

import pytest
import torch

from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.processing import (
    InvertibleMixin,
    Processor,
    Recipe,
    Standardize,
    StypeDispatch,
)


class _RecordingModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables: ClassVar[bool] = True
    supports_vectorized_ensemble: ClassVar[bool] = True

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
        table = x_query if x_query is not None else x_context
        assert table is not None
        numerical = table.numerical
        if x_query is not None and related_query_tables is not None:
            numerical = (
                numerical
                + related_query_tables.tables["users"].numerical[..., :1]
            )
        return TableTensor.from_tensor(numerical)

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()


class _UnsupportedRecordingModel(_RecordingModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.numerical, Stype.categorical})
    supports_related_tables = False


class _GeneratorRecordingProcessor(Processor, InvertibleMixin):
    supported_stypes = frozenset(Stype)
    member_specific_fit = True
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
        features=StypeDispatch(numerical=Standardize()),
    )


def _generator_recipe() -> Recipe:
    return Recipe(
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

    return list(_GeneratorRecordingProcessor.draws)


@pytest.mark.parametrize("cached", [False, True])
def test_model_recipe_fitting_honors_generator(cached: bool) -> None:
    first = _fit_draws(seed=0, cached=cached)
    second = _fit_draws(seed=0, cached=cached)
    different_seed = _fit_draws(seed=1, cached=cached)

    assert first
    assert len(first) == len(second) == len(different_seed)
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
    torch.testing.assert_close(
        direct.numerical,
        torch.tensor([[[5.0]], [[5.0]]]),
    )

    model.fit(
        x_context,
        y_context,
        related_context,
        recipe=_recipe(),
        num_estimators=2,
    )
    prediction = model.predict(x_query, related_query)
    repeated = model.predict(x_query, related_query)

    torch.testing.assert_close(prediction.numerical, direct.numerical)
    torch.testing.assert_close(
        repeated.numerical,
        prediction.numerical,
        rtol=0,
        atol=0,
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
