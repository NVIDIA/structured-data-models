from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.processing import (
    AlignCategories,
    Choice,
    Clip,
    DropConstantColumns,
    EnsembleRelatedTables,
    EnsembleTable,
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


class _VectorizedFirstFeatureModel(_FirstFeatureModel):
    def _forward_ensemble_group(
        self,
        *,
        members: tuple[int, ...],
        x_context: EnsembleTable | None,
        y_context: EnsembleTable | None,
        x_query: EnsembleTable | None,
        related_context_tables: EnsembleRelatedTables | None,
        related_query_tables: EnsembleRelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        kwargs: dict[str, Any],
    ) -> tuple[TableTensor, ...]:
        output = self._forward(
            x_context=(
                x_context.materialize(members)
                if x_context is not None
                else None
            ),
            y_context=(
                y_context.materialize(members)
                if y_context is not None
                else None
            ),
            x_query=(
                x_query.materialize(members) if x_query is not None else None
            ),
            related_context_tables=(
                related_context_tables.materialize(members)
                if related_context_tables is not None
                else None
            ),
            related_query_tables=(
                related_query_tables.materialize(members)
                if related_query_tables is not None
                else None
            ),
            cache=cache,
            generator=generator,
            **kwargs,
        )
        return tuple(output[index] for index in range(len(members)))


class _ClassCountModel(ICLModel):
    supported_feature_stypes = frozenset({Stype.numerical})
    supported_target_stypes = frozenset({Stype.categorical})
    supports_related_tables = False

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


class _ParallelOOMModel(_VectorizedFirstFeatureModel):
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


class _Float32Model(_FirstFeatureModel):
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
        del y_context, related_context_tables, related_query_tables, kwargs
        table = x_query if x_query is not None else x_context
        assert table is not None
        if cache is not None and cache.is_recording:
            cache["draw"] = torch.rand(
                (), device=table.device, generator=generator
            )
            return TableTensor.from_tensor(table.numerical[..., :0, :1])
        draw = (
            torch.rand((), device=table.device, generator=generator)
            if cache is None
            else cast(torch.Tensor, cache["draw"])
        )
        return TableTensor.from_tensor(
            torch.zeros_like(table.numerical[..., :1]) + draw
        )


class _VectorizedRandomOutputModel(_VectorizedFirstFeatureModel):
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


def test_recipe_preserves_implicit_regression_target_decode() -> None:
    x_context = torch.tensor([[0.0], [2.0], [4.0]])
    y_context = torch.tensor([[10.0], [12.0], [14.0]])

    actual = _FirstFeatureModel()(
        x_context,
        y_context,
        torch.tensor([[1.0]]),
        recipe=Recipe(
            target=Standardize(),
            output=ReduceEstimators(),
        ),
        num_estimators=2,
        ensemble_mode="parallel",
    )

    scale = y_context.std(dim=0, correction=0)
    expected = y_context.mean(dim=0) + scale
    torch.testing.assert_close(actual.numerical.squeeze(0), expected)


def test_parallel_and_sequential_execution_are_equivalent() -> None:
    args = (
        torch.tensor([[0.0], [2.0], [4.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0], [3.0]]),
    )

    parallel = _VectorizedFirstFeatureModel()(
        *args,
        recipe=_recipe(),
        num_estimators=8,
        ensemble_mode="parallel",
    )
    sequential = _VectorizedFirstFeatureModel()(
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


@pytest.mark.parametrize("cached", [False, True])
def test_member_rng_order_survives_interleaved_schemas(cached: bool) -> None:
    args = (
        torch.tensor([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0, 1.0], [3.0, 1.0]]),
    )
    recipe = Recipe(
        features=Choice(
            Identity(),
            DropConstantColumns(),
            selection="round_robin",
        )
    )
    parallel_model = _RandomOutputModel()
    sequential_model = _RandomOutputModel()
    parallel_generator = torch.Generator().manual_seed(7)
    sequential_generator = torch.Generator().manual_seed(7)

    if cached:
        parallel_model.fit(
            *args[:2],
            recipe=recipe,
            num_estimators=4,
            ensemble_mode="parallel",
            generator=parallel_generator,
        )
        sequential_model.fit(
            *args[:2],
            recipe=recipe,
            num_estimators=4,
            ensemble_mode="sequential",
            generator=sequential_generator,
        )
        parallel = parallel_model.predict(args[2])
        sequential = sequential_model.predict(args[2])
    else:
        parallel = parallel_model(
            *args,
            recipe=recipe,
            num_estimators=4,
            ensemble_mode="parallel",
            generator=parallel_generator,
        )
        sequential = sequential_model(
            *args,
            recipe=recipe,
            num_estimators=4,
            ensemble_mode="sequential",
            generator=sequential_generator,
        )

    torch.testing.assert_close(parallel.numerical, sequential.numerical)
    assert torch.equal(
        parallel_generator.get_state(),
        sequential_generator.get_state(),
    )


def test_vectorized_model_rng_is_memberwise_equivalent() -> None:
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


def test_cached_output_preserves_query_dtype() -> None:
    model = _Float32Model()
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


@pytest.mark.parametrize("explicit_decode", [False, True])
def test_classification_outputs_are_remapped_before_reduction(
    explicit_decode: bool,
) -> None:
    target = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [0], [1], [2]]),
            categories=(torch.tensor([10, 20, 30]),),
        ),
    )
    output = (
        (TargetDecode(), ReduceEstimators())
        if explicit_decode
        else ReduceEstimators()
    )
    recipe = Recipe(
        target=(AlignCategories(), ShuffleCategories(method="shift")),
        output=output,
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


@pytest.mark.parametrize("cached", [False, True])
def test_auto_mode_restores_global_rng_before_sequential_retry(
    cached: bool,
) -> None:
    args = (
        torch.tensor([[0.0], [2.0], [4.0]]),
        torch.tensor([[10.0], [12.0], [14.0]]),
        torch.tensor([[1.0], [3.0]]),
    )

    torch.manual_seed(5)
    auto_model = _ParallelOOMModel()
    if cached:
        auto_model.fit(
            *args[:2],
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="auto",
        )
        actual = auto_model.predict(args[2])
    else:
        actual = auto_model(
            *args,
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="auto",
        )
    actual_rng_state = torch.random.get_rng_state()

    torch.manual_seed(5)
    sequential_model = _ParallelOOMModel()
    if cached:
        sequential_model.fit(
            *args[:2],
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="sequential",
        )
        expected = sequential_model.predict(args[2])
    else:
        expected = sequential_model(
            *args,
            recipe=_recipe(),
            num_estimators=8,
            ensemble_mode="sequential",
        )

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert torch.equal(actual_rng_state, torch.random.get_rng_state())


def test_invalid_ensemble_mode_fails_clearly() -> None:
    with pytest.raises(ValueError, match="ensemble_mode"):
        _FirstFeatureModel()(
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[2.0]]),
            num_estimators=2,
            ensemble_mode="unknown",
        )
