from __future__ import annotations

from typing import Any, ClassVar

import pytest
import torch

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.processing import (
    AlignCategories,
    Choice,
    Clip,
    Identity,
    Recipe,
    ReduceEstimators,
    ShuffleCategories,
    Standardize,
    TargetDecode,
)


class _FirstFeatureModel(ICLModel):
    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_related_tables: ClassVar[bool] = False
    supports_vectorized_ensemble: ClassVar[bool] = True

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()

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
        del y_context, related_context_tables, related_query_tables
        del generator, kwargs
        if cache is not None and cache.is_recording:
            assert x_context is not None
            cache["value"] = x_context.numerical[..., -1:, :1]
            return TableTensor.from_tensor(x_context.numerical[..., :0, :1])
        assert x_query is not None
        return TableTensor.from_tensor(x_query.numerical[..., :1])


class _ClassCountModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.categorical})
    supports_related_tables = False
    supports_vectorized_ensemble = True

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()

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
        del x_context, related_context_tables, related_query_tables
        del cache, generator, kwargs
        assert y_context is not None
        assert x_query is not None
        codes = y_context.categorical.code.squeeze(-1).to(torch.long)
        num_classes = y_context.categorical.categories[0].numel()
        observed = codes >= 0
        counts = torch.nn.functional.one_hot(
            codes.clamp_min(0),
            num_classes=num_classes,
        )
        counts = (counts * observed.unsqueeze(-1)).sum(dim=-2)
        numerical = counts.to(x_query.numerical.dtype).unsqueeze(-2)
        numerical = numerical.expand(
            *numerical.shape[:-2],
            x_query.size(-2),
            num_classes,
        )
        return TableTensor.from_tensor(numerical)


class _ParallelOOMModel(_FirstFeatureModel):
    supports_vectorized_ensemble_rng = True

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
        if generator is not None:
            torch.rand((), device=table.device, generator=generator)
        if table.size(0) > 1:
            raise torch.cuda.OutOfMemoryError("test parallel OOM")
        return super()._forward(
            x_context,
            y_context,
            x_query,
            related_context_tables,
            related_query_tables,
            cache,
            generator,
            **kwargs,
        )


class _Float32SequentialModel(_FirstFeatureModel):
    supports_vectorized_ensemble = False

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
        output = super()._forward(
            x_context,
            y_context,
            x_query,
            related_context_tables,
            related_query_tables,
            cache,
            generator,
            **kwargs,
        )
        return TableTensor.from_tensor(output.numerical.to(torch.float32))


class _RandomOutputModel(_FirstFeatureModel):
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
        del x_context, y_context, related_context_tables, related_query_tables
        del cache, kwargs
        assert x_query is not None
        draw = torch.rand((), device=x_query.device, generator=generator)
        return TableTensor.from_tensor(
            torch.zeros_like(x_query.numerical[..., :1]) + draw
        )


class _VectorizedRandomOutputModel(_FirstFeatureModel):
    supports_vectorized_ensemble_rng = True

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
        del x_context, y_context, related_context_tables, related_query_tables
        del cache, kwargs
        assert x_query is not None
        draws = torch.rand(
            (x_query.size(0), 1, 1),
            device=x_query.device,
            generator=generator,
        )
        return TableTensor.from_tensor(
            torch.zeros_like(x_query.numerical[..., :1]) + draws
        )


def _recipe() -> Recipe:
    return Recipe(
        features=Choice(
            Identity(),
            Clip(min_value=0.0, max_value=0.0),
            selection="round_robin",
        ),
        target=Standardize(),
        output=(TargetDecode(), ReduceEstimators()),
    )


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
def test_regression_target_decode_precedes_estimator_reduction(
    mode: str,
) -> None:
    model = _FirstFeatureModel()
    x_context = torch.tensor([[0.0], [2.0], [4.0]])
    y_context = torch.tensor([[10.0], [12.0], [14.0]])
    x_query = torch.tensor([[1.0]])

    actual = model(
        x_context,
        y_context,
        x_query,
        recipe=_recipe(),
        num_estimators=2,
        ensemble_mode=mode,
    )

    # Raw member predictions are 1 and 0. Inverting Standardize first maps
    # them to mean + scale and mean; reduction therefore happens afterwards.
    scale = y_context.std(dim=0, correction=0)
    expected = y_context.mean(dim=0) + scale / 2
    torch.testing.assert_close(actual.numerical.squeeze(0), expected)


def test_parallel_and_sequential_execution_are_equivalent() -> None:
    args = (
        torch.tensor([[0.0], [2.0], [4.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0], [3.0]]),
    )

    parallel = _FirstFeatureModel()(
        *args,
        recipe=_recipe(),
        num_estimators=8,
        ensemble_mode="parallel",
    )
    sequential = _FirstFeatureModel()(
        *args,
        recipe=_recipe(),
        num_estimators=8,
        ensemble_mode="sequential",
    )

    torch.testing.assert_close(
        parallel.numerical,
        sequential.numerical,
        rtol=0,
        atol=0,
    )


def test_parallel_and_sequential_model_rng_are_equivalent() -> None:
    args = (
        torch.tensor([[0.0], [2.0], [4.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0], [3.0]]),
    )
    recipe = Recipe(output=(TargetDecode(), ReduceEstimators()))

    parallel = _RandomOutputModel()(
        *args,
        recipe=recipe,
        num_estimators=8,
        ensemble_mode="parallel",
        generator=torch.Generator().manual_seed(7),
    )
    sequential = _RandomOutputModel()(
        *args,
        recipe=recipe,
        num_estimators=8,
        ensemble_mode="sequential",
        generator=torch.Generator().manual_seed(7),
    )

    torch.testing.assert_close(parallel.numerical, sequential.numerical)


def test_opted_in_vectorized_model_rng_is_memberwise_equivalent() -> None:
    args = (
        torch.tensor([[0.0], [2.0], [4.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0], [3.0]]),
    )
    parallel_generator = torch.Generator().manual_seed(7)
    sequential_generator = torch.Generator().manual_seed(7)

    parallel = _VectorizedRandomOutputModel()(
        *args,
        recipe=Recipe(),
        num_estimators=8,
        ensemble_mode="parallel",
        generator=parallel_generator,
    )
    sequential = _VectorizedRandomOutputModel()(
        *args,
        recipe=Recipe(),
        num_estimators=8,
        ensemble_mode="sequential",
        generator=sequential_generator,
    )

    torch.testing.assert_close(parallel.numerical, sequential.numerical)
    assert torch.equal(
        parallel_generator.get_state(),
        sequential_generator.get_state(),
    )


def test_fit_predict_reuses_the_same_ensemble_plan() -> None:
    x_context = torch.tensor([[0.0], [2.0], [4.0]])
    y_context = torch.tensor([[10.0], [12.0], [14.0]])
    x_query = torch.tensor([[1.0], [3.0]])
    direct_model = _FirstFeatureModel()
    cached_model = _FirstFeatureModel()

    direct = direct_model(
        x_context,
        y_context,
        x_query,
        recipe=_recipe(),
        num_estimators=8,
        ensemble_mode="sequential",
    )
    cached_model.fit(
        x_context,
        y_context,
        recipe=_recipe(),
        num_estimators=8,
        ensemble_mode="sequential",
    )
    first = cached_model.predict(x_query)
    repeated = cached_model.predict(x_query)

    torch.testing.assert_close(first.numerical, direct.numerical)
    torch.testing.assert_close(
        repeated.numerical,
        first.numerical,
        rtol=0,
        atol=0,
    )


def test_non_vectorized_cached_output_preserves_query_dtype() -> None:
    model = _Float32SequentialModel()
    x_context = torch.tensor([[0.0], [2.0], [4.0]], dtype=torch.float64)
    y_context = torch.tensor([[10.0], [12.0], [14.0]], dtype=torch.float64)
    x_query = torch.tensor([[1.0], [3.0]], dtype=torch.float64)

    direct = model(
        x_context,
        y_context,
        x_query,
        recipe=Recipe(),
        num_estimators=2,
        ensemble_mode="sequential",
    )
    model.fit(
        x_context,
        y_context,
        recipe=Recipe(),
        num_estimators=2,
        ensemble_mode="sequential",
    )
    cached = model.predict(x_query)

    assert direct.numerical.dtype == torch.float64
    assert cached.numerical.dtype == torch.float64
    torch.testing.assert_close(cached.numerical, direct.numerical)


def test_classification_outputs_are_remapped_before_reduction() -> None:
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [0], [1], [2]]),
            categories=(torch.tensor([10, 20, 30]),),
        ),
    )
    recipe = Recipe(
        target=(AlignCategories(), ShuffleCategories(method="shift")),
        output=(TargetDecode(), ReduceEstimators()),
    )

    actual = _ClassCountModel()(
        torch.arange(4, dtype=torch.float32).unsqueeze(-1),
        target,
        torch.tensor([[5.0], [6.0]]),
        recipe=recipe,
        num_estimators=8,
        ensemble_mode="parallel",
        generator=torch.Generator().manual_seed(11),
    )

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[2.0, 1.0, 1.0], [2.0, 1.0, 1.0]]),
    )


@pytest.mark.parametrize("cached", [False, True])
def test_auto_mode_retries_parallel_oom_sequentially(cached: bool) -> None:
    args = (
        torch.tensor([[0.0], [2.0], [4.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0], [3.0]]),
    )
    auto_model = _ParallelOOMModel()
    sequential_model = _ParallelOOMModel()
    auto_generator = torch.Generator().manual_seed(5)
    sequential_generator = torch.Generator().manual_seed(5)

    if cached:
        auto_model.fit(
            *args[:2],
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="auto",
            generator=auto_generator,
        )
        sequential_model.fit(
            *args[:2],
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="sequential",
            generator=sequential_generator,
        )
        actual = auto_model.predict(args[2])
        expected = sequential_model.predict(args[2])
    else:
        actual = auto_model(
            *args,
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="auto",
            generator=auto_generator,
        )
        expected = sequential_model(
            *args,
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="sequential",
            generator=sequential_generator,
        )

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert torch.equal(
        auto_generator.get_state(), sequential_generator.get_state()
    )


def test_invalid_ensemble_mode_fails_clearly() -> None:
    with pytest.raises(ValueError, match="ensemble_mode"):
        _FirstFeatureModel()(
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[2.0]]),
            num_estimators=2,
            ensemble_mode="unknown",
        )
