from typing import ClassVar, cast

import pytest
import torch
from sdm import (
    CategoricalTensor,
    RelatedTables,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.cache import Cache
from sdm.models import Model
from sdm.processing import (
    Clip,
    Identity,
    InvertibleMixin,
    MeanImpute,
    Processor,
    Recipe,
    ReduceEstimators,
    SoftmaxTemperature,
    StandardScale,
)
from sdm.testing import withCUDA
from torch import Tensor


class _CyclingCategoryShuffle(Processor):
    supported_stypes = frozenset({Stype.categorical})
    plans: ClassVar[tuple[tuple[int, ...], ...]] = (
        (0, 1, 2),
        (1, 2, 0),
    )
    next_plan: ClassVar[int] = 0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("permutation", torch.empty(0, dtype=torch.long))

    @classmethod
    def reset(cls) -> None:
        cls.next_plan = 0

    def _fit(self, table: TableTensor) -> None:
        plan = self.plans[self.__class__.next_plan]
        self.__class__.next_plan += 1
        self.permutation = torch.tensor(plan, device=table.device)

    def _transform(self, table: TableTensor) -> TableTensor:
        data = table.categorical.as_tensor()
        transformed = self.permutation[data.to(torch.long)].to(data.dtype)
        categories = table.categorical.categories[0]
        return table.replace_blocks(
            categorical=CategoricalTensor(
                data=transformed,
                categories=(categories[self.permutation.argsort()],),
            )
        )


class _LogTarget(Processor, InvertibleMixin):
    supported_stypes = frozenset({Stype.numerical})

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.log())

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.exp())


class _SquareOutput(Processor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.square())


class _SwapFeatures(Processor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = tuple(reversed(table.columns[Stype.numerical]))
        return TableTensor.from_tensor(
            table.numerical.flip(-1),
            columns=columns,
        )


class _AnalyticModel(Model):
    supports_related_tables = False

    def __init__(self, outputs: Tensor) -> None:
        super().__init__()
        self.register_buffer("_outputs_buffer", outputs)
        self._uncached_calls = 0
        self._cache_build_calls = 0
        self.model_targets: list[Tensor] = []

    @property
    def outputs(self) -> Tensor:
        output = self._buffers["_outputs_buffer"]
        assert output is not None
        return output

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
    ) -> TableTensor:
        del related_context_tables, related_query_tables

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            target = y_context.categorical.as_tensor().squeeze(-1)
            classes = y_context.categorical.categories[0]
            self.model_targets.append(target.clone())
        elif y_context is not None:
            self.model_targets.append(y_context.numerical.squeeze(-1).clone())
        elif cache is not None:
            classes = cast(Tensor | None, cache["classes"])

        if cache is not None and cache.is_recording:
            member = self._cache_build_calls
            self._cache_build_calls += 1
            cache["member"] = member
        elif cache is None:
            member = self._uncached_calls % self.outputs.size(0)
            self._uncached_calls += 1
        else:
            member = cast(int, cache["member"])

        if x_query is None:
            values = self.outputs.new_empty((0, self.outputs.size(-1)))
        else:
            values = self.outputs[member].expand(x_query.size(-2), -1)

        columns = (
            tuple(str(value) for value in classes.tolist())
            if classes is not None
            else tuple(f"output_{index}" for index in range(values.size(-1)))
        )
        return TableTensor.from_tensor(values, columns=columns)


class _FeatureEchoModel(Model):
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
    ) -> TableTensor:
        del x_context, y_context, related_context_tables, related_query_tables
        assert cache is None
        assert x_query is not None
        return TableTensor.from_tensor(
            x_query.numerical[..., :1],
            columns=("prediction",),
        )


def _features(device: torch.device) -> TableTensor:
    return TableTensor.from_tensor(
        torch.ones(4, 1, device=device),
        columns=("x",),
    )


def _classification_target(device: torch.device) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("label",)},
        categorical=CategoricalTensor(
            data=torch.tensor(
                [[0], [1], [2]],
                dtype=torch.int64,
                device=device,
            ),
            categories=(
                StringTensor.from_list(["red", "green", "blue"]).to(device),
            ),
        ),
    )


def _classification_recipe() -> Recipe:
    return Recipe(
        target=[_CyclingCategoryShuffle()],
        output=[
            ReduceEstimators(),
            SoftmaxTemperature(temperature=0.9),
        ],
    )


def _classification_outputs(device: torch.device) -> tuple[Tensor, Tensor]:
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        device=device,
    )
    permutations = torch.tensor(
        _CyclingCategoryShuffle.plans,
        device=device,
    )
    raw = torch.stack(
        [
            canonical[index].index_select(
                0,
                permutations[index].argsort(),
            )
            for index in range(2)
        ]
    )
    return canonical, raw


@withCUDA
def test_category_columns_map_members_before_logit_aggregation(
    device: torch.device,
) -> None:
    _CyclingCategoryShuffle.reset()
    canonical, raw = _classification_outputs(device)
    model = _AnalyticModel(raw)
    features = _features(device)

    actual = model(
        features[:3],
        _classification_target(device),
        features[3:],
        recipe=_classification_recipe(),
        num_estimators=2,
    )

    expected = (canonical.mean(dim=0) / 0.9).softmax(dim=-1).unsqueeze(0)
    probability_average = (
        (canonical / 0.9).softmax(dim=-1).mean(dim=0).unsqueeze(0)
    )
    torch.testing.assert_close(actual.numerical, expected, rtol=0, atol=1e-7)
    assert not torch.allclose(actual.numerical, probability_average)
    assert torch.equal(
        model.model_targets[0],
        torch.tensor([0, 1, 2], device=device),
    )
    assert torch.equal(
        model.model_targets[1],
        torch.tensor([1, 2, 0], device=device),
    )


@withCUDA
def test_regression_target_inverse_runs_per_member_before_aggregation(
    device: torch.device,
) -> None:
    outputs = torch.tensor(
        [[0.0], [torch.log(torch.tensor(4.0))]],
        device=device,
    )
    model = _AnalyticModel(outputs)
    target = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0], [4.0]], device=device)
    )
    features = _features(device)

    actual = model(
        features[:3],
        target,
        features[3:],
        recipe=Recipe(
            target=[_LogTarget()],
            output=[ReduceEstimators(), Identity()],
        ),
        num_estimators=2,
    )

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[2.5]], device=device),
    )
    inverse_after_average = outputs.mean(dim=0).exp().unsqueeze(0)
    assert not torch.equal(actual.numerical, inverse_after_average)


@withCUDA
def test_output_transform_runs_once_after_aggregation(
    device: torch.device,
) -> None:
    model = _AnalyticModel(torch.tensor([[1.0], [3.0]], device=device))
    target = TableTensor.from_tensor(torch.ones(3, 1, device=device))
    features = _features(device)

    actual = model(
        features[:3],
        target,
        features[3:],
        recipe=Recipe(
            target=[Identity()],
            output=[ReduceEstimators(), _SquareOutput()],
        ),
        num_estimators=2,
    )

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[4.0]], device=device),
    )
    assert not torch.equal(
        actual.numerical,
        torch.tensor([[5.0]], device=device),
    )


@withCUDA
def test_deterministic_model_end_to_end_includes_pre_and_postprocessing(
    device: torch.device,
) -> None:
    features = TableTensor.from_tensor(
        torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [1_000.0, 40.0]],
            device=device,
        ),
        columns=("a", "b"),
    )
    target_values = torch.tensor([[1.0], [2.0], [3.0]], device=device)
    target = TableTensor.from_tensor(target_values)
    recipe = Recipe(
        features=[
            MeanImpute(),
            StandardScale(),
            Clip(min_value=-2.0, max_value=2.0),
            _SwapFeatures(),
        ],
        target=[StandardScale()],
        output=[ReduceEstimators(), _SquareOutput()],
    )

    actual = _FeatureEchoModel()(
        features[:3],
        target,
        features[3:],
        recipe=recipe,
    )

    target_scale = target_values.squeeze(-1).std(correction=0)
    canonical = 2.0 * target_scale + 2.0
    expected = canonical.square().reshape(1, 1)
    torch.testing.assert_close(actual.numerical, expected)


@withCUDA
def test_cache_reuses_member_recipe_and_category_mapping(
    device: torch.device,
) -> None:
    canonical, raw = _classification_outputs(device)
    features = _features(device)
    target = _classification_target(device)

    _CyclingCategoryShuffle.reset()
    direct_model = _AnalyticModel(raw)
    direct = direct_model(
        features[:3],
        target,
        features[3:],
        recipe=_classification_recipe(),
        num_estimators=2,
    )

    _CyclingCategoryShuffle.reset()
    cached_model = _AnalyticModel(raw)
    recipe = _classification_recipe()
    cached_model.fit(
        features[:3],
        target,
        recipe=recipe,
        num_estimators=2,
    )
    assert cached_model._caches is not None
    cached_recipes = [cache["recipe"] for cache in cached_model._caches]
    assert cached_recipes[0] is not recipe
    assert cached_recipes[1] is not recipe
    assert cached_recipes[0] is not cached_recipes[1]
    cached = cached_model.predict(features[3:])
    repeated = cached_model.predict(features[3:])

    expected = (canonical.mean(dim=0) / 0.9).softmax(dim=-1).unsqueeze(0)
    torch.testing.assert_close(direct.numerical, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        cached.numerical, direct.numerical, rtol=0, atol=0
    )
    torch.testing.assert_close(
        repeated.numerical,
        cached.numerical,
        rtol=0,
        atol=0,
    )


@pytest.mark.xfail(
    strict=True,
    reason="Model does not yet validate num_estimators before stacking",
)
def test_num_estimators_must_be_positive() -> None:
    model = _AnalyticModel(torch.tensor([[1.0]]))
    target = TableTensor.from_tensor(torch.ones(3, 1))
    features = _features(torch.device("cpu"))

    with pytest.raises(ValueError, match="num_estimators must be positive"):
        model(
            features[:3],
            target,
            features[3:],
            num_estimators=0,
        )
