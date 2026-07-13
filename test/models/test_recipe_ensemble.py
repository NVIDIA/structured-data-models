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
    HardClip,
    Identity,
    InvertibleMixin,
    MeanImpute,
    Processor,
    Recipe,
    SoftmaxTemperature,
    StandardScale,
)
from torch import Tensor


class _CyclingClassShuffle(Processor):
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

    def _fit(self, inp: TableTensor) -> None:
        plan = self.plans[self.__class__.next_plan]
        self.__class__.next_plan += 1
        self.permutation = torch.tensor(plan, device=inp.device)

    def _transform(self, inp: TableTensor) -> TableTensor:
        data = inp.categorical.as_tensor()
        transformed = self.permutation[data.to(torch.long)].to(data.dtype)
        categories = inp.categorical.categories[0]
        return inp.replace_blocks(
            categorical=CategoricalTensor(
                data=transformed,
                categories=(categories[self.permutation.argsort()],),
            )
        )


class _LogTarget(Processor, InvertibleMixin):
    supported_stypes = frozenset({Stype.numerical})

    def _transform(self, inp: TableTensor) -> TableTensor:
        return inp.replace_blocks(numerical=inp.numerical.log())

    def _inverse_transform(self, inp: TableTensor) -> TableTensor:
        return inp.replace_blocks(numerical=inp.numerical.exp())


class _SquareOutput(Processor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, inp: TableTensor) -> TableTensor:
        return inp.replace_blocks(numerical=inp.numerical.square())


class _SwapFeatures(Processor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, inp: TableTensor) -> TableTensor:
        columns = tuple(reversed(inp.columns[Stype.numerical]))
        return TableTensor.from_tensor(
            inp.numerical.flip(-1),
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
        x: Tensor,
        y: Tensor,
        related_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> Tensor:
        del related_tables
        self.model_targets.append(y.clone())
        if cache is not None and cache.is_recording:
            member = self._cache_build_calls
            self._cache_build_calls += 1
            cache["member"] = member
            return x.new_empty((0, self.outputs.size(-1)))

        if cache is None:
            member = self._uncached_calls % self.outputs.size(0)
            self._uncached_calls += 1
            n_test = x.size(-2) - y.size(-1)
        else:
            member = cast(int, cache["member"])
            n_test = x.size(-2)
        return self.outputs[member].to(x).expand(n_test, -1)


class _FeatureEchoModel(Model):
    supports_related_tables = False

    @classmethod
    def default_recipe(cls) -> Recipe:
        return Recipe()

    def _forward(
        self,
        x: Tensor,
        y: Tensor,
        related_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> Tensor:
        del related_tables
        assert cache is None
        return x[..., y.size(-1) :, :1]


def _features() -> TableTensor:
    return TableTensor.from_tensor(torch.ones(4, 1), columns=("x",))


def _classification_target() -> TableTensor:
    return TableTensor(
        columns={"categorical": ("label",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [2]], dtype=torch.int64),
            categories=(StringTensor.from_list(["red", "green", "blue"]),),
        ),
    )


def _classification_recipe() -> Recipe:
    return Recipe(
        target=[_CyclingClassShuffle()],
        output=[SoftmaxTemperature(temperature=0.9)],
    )


def _classification_outputs() -> tuple[Tensor, Tensor]:
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    permutations = torch.tensor(_CyclingClassShuffle.plans)
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


def test_category_columns_map_members_before_logit_aggregation() -> None:
    _CyclingClassShuffle.reset()
    canonical, raw = _classification_outputs()
    model = _AnalyticModel(raw)

    actual = model(
        _features(),
        _classification_target(),
        recipe=_classification_recipe(),
        num_estimators=2,
    )

    expected = (canonical.mean(dim=0) / 0.9).softmax(dim=-1).unsqueeze(0)
    probability_average = (
        (canonical / 0.9).softmax(dim=-1).mean(dim=0).unsqueeze(0)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    assert not torch.allclose(actual, probability_average)
    assert torch.equal(model.model_targets[0], torch.tensor([0, 1, 2]))
    assert torch.equal(model.model_targets[1], torch.tensor([1, 2, 0]))


def test_regression_target_inverse_runs_per_member_before_aggregation() -> (
    None
):
    outputs = torch.tensor([[0.0], [torch.log(torch.tensor(4.0))]])
    model = _AnalyticModel(outputs)
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))

    actual = model(
        _features(),
        target,
        recipe=Recipe(target=[_LogTarget()], output=[Identity()]),
        num_estimators=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[2.5]]))
    inverse_after_average = outputs.mean(dim=0).exp().unsqueeze(0)
    assert not torch.equal(actual, inverse_after_average)


def test_output_transform_runs_once_after_aggregation() -> None:
    model = _AnalyticModel(torch.tensor([[1.0], [3.0]]))
    target = TableTensor.from_tensor(torch.ones(3, 1))

    actual = model(
        _features(),
        target,
        recipe=Recipe(target=[Identity()], output=[_SquareOutput()]),
        num_estimators=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[4.0]]))
    assert not torch.equal(actual, torch.tensor([[5.0]]))


def test_deterministic_model_end_to_end_includes_pre_and_postprocessing() -> (
    None
):
    features = TableTensor.from_tensor(
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [1_000.0, 40.0]]),
        columns=("a", "b"),
    )
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [3.0]]))
    recipe = Recipe(
        features=[
            MeanImpute(),
            StandardScale(),
            HardClip(min_value=-2.0, max_value=2.0),
            _SwapFeatures(),
        ],
        target=[StandardScale()],
        output=[_SquareOutput()],
    )

    actual = _FeatureEchoModel()(features, target, recipe=recipe)

    target_scale = torch.tensor([1.0, 2.0, 3.0]).std(correction=0)
    canonical = 2.0 * target_scale + 2.0
    expected = canonical.square().reshape(1, 1)
    torch.testing.assert_close(actual, expected)


def test_cache_reuses_member_recipe_and_category_mapping() -> None:
    canonical, raw = _classification_outputs()
    features = _features()

    _CyclingClassShuffle.reset()
    direct_model = _AnalyticModel(raw)
    direct = direct_model(
        features,
        _classification_target(),
        recipe=_classification_recipe(),
        num_estimators=2,
    )

    _CyclingClassShuffle.reset()
    cached_model = _AnalyticModel(raw)
    recipe = _classification_recipe()
    cached_model.fit(
        features[:3],
        _classification_target(),
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
    torch.testing.assert_close(direct, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(cached, direct, rtol=0, atol=0)
    torch.testing.assert_close(repeated, cached, rtol=0, atol=0)


def test_num_estimators_must_be_positive() -> None:
    model = _AnalyticModel(torch.tensor([[1.0]]))
    target = TableTensor.from_tensor(torch.ones(3, 1))

    with pytest.raises(ValueError, match="num_estimators must be positive"):
        model(_features(), target, num_estimators=0)
