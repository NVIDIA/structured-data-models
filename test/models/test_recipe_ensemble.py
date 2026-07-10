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
from sdm.models import BaseModel
from sdm.processing import (
    Identity,
    InvertibleMixin,
    Processor,
    Recipe,
    SoftmaxTemperature,
)
from torch import Tensor


class _CyclingFeaturePermute(Processor):
    supported_stypes = frozenset({Stype.numerical})

    plans: ClassVar[tuple[tuple[int, ...], ...]] = (
        (0, 1, 2),
        (1, 2, 0),
    )
    next_plan: ClassVar[int] = 0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "permutation",
            torch.empty(0, dtype=torch.long),
        )

    @classmethod
    def reset(cls) -> None:
        cls.next_plan = 0

    def _fit(self, input: TableTensor) -> None:
        plan = self.plans[self.__class__.next_plan]
        self.__class__.next_plan += 1
        self.permutation = torch.tensor(plan, device=input.device)

    def _transform(self, input: TableTensor) -> TableTensor:
        columns = input.columns[Stype.numerical]
        indices = self.permutation.tolist()
        return TableTensor.from_tensor(
            input.numerical.index_select(-1, self.permutation),
            columns=[columns[index] for index in indices],
        )


class _CyclingCategoryShuffle(Processor, InvertibleMixin):
    supported_stypes = frozenset({Stype.categorical})

    plans: ClassVar[tuple[tuple[int, ...], ...]] = (
        (0, 1, 2),
        (1, 2, 0),
    )
    next_plan: ClassVar[int] = 0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "permutation",
            torch.empty(0, dtype=torch.long),
        )

    @classmethod
    def reset(cls) -> None:
        cls.next_plan = 0

    def _fit(self, input: TableTensor) -> None:
        plan = self.plans[self.__class__.next_plan]
        self.__class__.next_plan += 1
        self.permutation = torch.tensor(plan, device=input.device)

    def _transform(self, input: TableTensor) -> TableTensor:
        data = input.categorical.as_tensor()
        shuffled = self.permutation[data.to(torch.long)].to(data.dtype)
        category = input.categorical.categories[0]
        return TableTensor(
            columns={"categorical": input.columns[Stype.categorical]},
            categorical=CategoricalTensor(
                data=shuffled,
                categories=(category[self.permutation.argsort()],),
            ),
        )

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return TableTensor.from_tensor(
            input.numerical.index_select(-1, self.permutation)
        )


class _LogTarget(Processor, InvertibleMixin):
    supported_stypes = frozenset({Stype.numerical})

    def _transform(self, input: TableTensor) -> TableTensor:
        return input.replace_blocks(numerical=input.numerical.log())

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return input.replace_blocks(numerical=input.numerical.exp())


class _SquareOutput(Processor):
    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, input: TableTensor) -> TableTensor:
        return input.replace_blocks(numerical=input.numerical.square())


class _AnalyticModel(BaseModel):
    supports_related_tables = False
    outputs: Tensor

    def __init__(self, outputs: Tensor) -> None:
        super().__init__()
        self.register_buffer("outputs", outputs)
        self._uncached_calls = 0
        self._cache_build_calls = 0
        self.model_inputs: list[tuple[Tensor, Tensor]] = []

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
        self.model_inputs.append((x.clone(), y.clone()))

        if cache is not None and cache.is_recording:
            member = self._cache_build_calls
            self._cache_build_calls += 1
            cache["member"] = member
            cache["feature_signature"] = x[0].clone()
            return x.new_empty((0, self.outputs.size(-1)))

        if cache is not None:
            member = cast(int, cache["member"])
            torch.testing.assert_close(
                x[0],
                cast(Tensor, cache["feature_signature"]),
                rtol=0,
                atol=0,
            )
            n_test = x.size(-2)
        else:
            member = self._uncached_calls % self.outputs.size(0)
            self._uncached_calls += 1
            n_test = x.size(-2) - y.size(-1)

        return self.outputs[member].to(x).expand(n_test, -1)


def _features() -> TableTensor:
    values = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
        ]
    )
    return TableTensor.from_tensor(values, columns=("a", "b", "c"))


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
        features=[_CyclingFeaturePermute()],
        target=[_CyclingCategoryShuffle()],
        output=[SoftmaxTemperature(temperature=0.9)],
    )


def test_class_inverse_precedes_logit_aggregation_and_softmax() -> None:
    _CyclingFeaturePermute.reset()
    _CyclingCategoryShuffle.reset()
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    permutations = torch.tensor(_CyclingCategoryShuffle.plans)
    raw = torch.stack(
        [
            canonical[index].index_select(
                0,
                permutations[index].argsort(),
            )
            for index in range(2)
        ]
    )
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

    expected_features = (
        torch.tensor([10.0, 20.0, 30.0]),
        torch.tensor([20.0, 30.0, 10.0]),
    )
    expected_targets = (
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 0]),
    )
    for index, (x_model, y_model) in enumerate(model.model_inputs):
        torch.testing.assert_close(
            x_model[0], expected_features[index], rtol=0, atol=0
        )
        assert torch.equal(y_model, expected_targets[index])


def test_nonlinear_output_runs_once_after_aggregation() -> None:
    model = _AnalyticModel(torch.tensor([[1.0], [3.0]]))
    features = TableTensor.from_tensor(torch.ones(4, 1))
    target = TableTensor.from_tensor(torch.ones(3, 1))
    recipe = Recipe(target=[Identity()], output=[_SquareOutput()])

    actual = model(
        features,
        target,
        recipe=recipe,
        num_estimators=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[4.0]]))
    assert not torch.equal(actual, torch.tensor([[5.0]]))


def test_nonlinear_target_inverse_runs_per_member_before_aggregation() -> None:
    outputs = torch.tensor([[0.0], [torch.log(torch.tensor(4.0))]])
    model = _AnalyticModel(outputs)
    features = TableTensor.from_tensor(torch.ones(4, 1))
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))
    recipe = Recipe(target=[_LogTarget()], output=[Identity()])

    actual = model(
        features,
        target,
        recipe=recipe,
        num_estimators=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[2.5]]))
    inverse_after_average = model.outputs.mean(dim=0).exp().unsqueeze(0)
    assert not torch.equal(actual, inverse_after_average)


def test_cache_reuses_the_matching_fitted_recipe_per_member() -> None:
    _CyclingFeaturePermute.reset()
    _CyclingCategoryShuffle.reset()
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    permutations = torch.tensor(_CyclingCategoryShuffle.plans)
    raw = torch.stack(
        [
            canonical[index].index_select(
                0,
                permutations[index].argsort(),
            )
            for index in range(2)
        ]
    )
    model = _AnalyticModel(raw)
    features = _features()

    model.fit(
        features[:3],
        _classification_target(),
        recipe=_classification_recipe(),
        num_estimators=2,
    )
    actual = model.predict(features[3:])

    expected = (canonical.mean(dim=0) / 0.9).softmax(dim=-1).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    repeated = model.predict(features[3:])
    torch.testing.assert_close(repeated, expected, rtol=0, atol=1e-7)


def test_cache_and_non_cache_orchestration_are_equivalent() -> None:
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    permutations = torch.tensor(_CyclingCategoryShuffle.plans)
    raw = torch.stack(
        [
            canonical[index].index_select(
                0,
                permutations[index].argsort(),
            )
            for index in range(2)
        ]
    )
    features = _features()

    _CyclingFeaturePermute.reset()
    _CyclingCategoryShuffle.reset()
    direct_model = _AnalyticModel(raw)
    direct = direct_model(
        features,
        _classification_target(),
        recipe=_classification_recipe(),
        num_estimators=2,
    )

    _CyclingFeaturePermute.reset()
    _CyclingCategoryShuffle.reset()
    cached_model = _AnalyticModel(raw)
    cached_model.fit(
        features[:3],
        _classification_target(),
        recipe=_classification_recipe(),
        num_estimators=2,
    )
    cached = cached_model.predict(features[3:])

    torch.testing.assert_close(cached, direct, rtol=0, atol=0)


def test_num_estimators_must_be_positive() -> None:
    model = _AnalyticModel(torch.tensor([[1.0]]))
    with pytest.raises(ValueError, match="num_estimators must be positive"):
        model(
            TableTensor.from_tensor(torch.ones(2, 1)),
            TableTensor.from_tensor(torch.ones(1, 1)),
            num_estimators=0,
        )
