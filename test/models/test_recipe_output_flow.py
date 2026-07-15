from typing import ClassVar, cast

import torch
from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import Model, TabICLv2
from sdm.models.kumorfm.model import KumoRFM
from sdm.processing import (
    ClassDecode,
    EstimatorMean,
    Identity,
    InvertibleMixin,
    Processor,
    QuantileDecode,
    Recipe,
    RecipeContext,
    Sequential,
    SoftmaxTemperature,
    TargetDecode,
    TaskDispatch,
)
from torch import Tensor


class _CyclingClassShuffle(Processor):
    supported_stypes = frozenset({Stype.categorical})
    plans: ClassVar[tuple[tuple[int, ...], ...]] = ()
    next_plan: ClassVar[int] = 0

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("permutation", torch.empty(0, dtype=torch.long))

    @classmethod
    def reset(cls, num_classes: int) -> None:
        cls.plans = (
            tuple(range(num_classes)),
            (*range(1, num_classes), 0),
        )
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


class _OrderedReduction(Processor):
    """Encode estimator order as first + 10 * second."""

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical.select(
            -3, 0
        ) + 10 * table.numerical.select(-3, 1)
        return TableTensor.from_tensor(numerical)


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
        return Recipe(output=[EstimatorMean()])

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
        if y_context is not None:
            if y_context.categorical.size(-1) > 0:
                target = y_context.categorical.as_tensor()
            else:
                target = y_context.numerical
            self.model_targets.append(target.clone())

        if cache is not None and cache.is_recording:
            member = self._cache_build_calls
            self._cache_build_calls += 1
            cache["member"] = member
            return TableTensor.from_tensor(
                self.outputs.new_empty((0, self.outputs.size(-1)))
            )

        if cache is None:
            member = self._uncached_calls % self.outputs.size(0)
            self._uncached_calls += 1
        else:
            member = cast(int, cache["member"])

        assert x_query is not None
        numerical = self.outputs[member].to(x_query.device)
        numerical = numerical.expand(x_query.size(-2), -1)
        return TableTensor.from_tensor(numerical)


def _features(rows: int) -> TableTensor:
    return TableTensor.from_tensor(torch.ones(rows, 1), columns=("x",))


def _classification_target(num_classes: int) -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(num_classes, dtype=torch.int64).unsqueeze(-1),
        columns=("label",),
    )


def _classification_recipe() -> Recipe:
    return Recipe(
        target=[_CyclingClassShuffle()],
        output=[
            TaskDispatch(
                classification=ClassDecode(),
                regression=TargetDecode(),
            ),
            EstimatorMean(),
            TaskDispatch(
                classification=SoftmaxTemperature(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )


def test_classification_is_canonicalized_before_reduction() -> None:
    num_classes = 3
    _CyclingClassShuffle.reset(num_classes)
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
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
    model = _AnalyticModel(raw)

    actual = model(
        _features(num_classes),
        _classification_target(num_classes),
        _features(1),
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
        model.model_targets[1].squeeze(-1),
        torch.tensor(_CyclingClassShuffle.plans[1]),
    )


def test_fitted_context_belongs_to_each_cached_estimator() -> None:
    num_classes = 3
    _CyclingClassShuffle.reset(num_classes)
    canonical = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
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
    model = _AnalyticModel(raw)

    model.fit(
        _features(num_classes),
        _classification_target(num_classes),
        recipe=_classification_recipe(),
        num_estimators=2,
    )

    assert model._caches is not None
    contexts = [
        cast(RecipeContext, cache["context"]) for cache in model._caches
    ]
    recipes = [cast(Recipe, cache["recipe"]) for cache in model._caches]
    assert [context.estimator_index for context in contexts] == [0, 1]
    assert [context.class_indices for context in contexts] == list(
        _CyclingClassShuffle.plans
    )
    assert recipes[0] is not recipes[1]

    actual = model.predict(_features(1))
    expected = (canonical.mean(dim=0) / 0.9).softmax(dim=-1).unsqueeze(0)
    torch.testing.assert_close(actual.numerical, expected, rtol=0, atol=1e-7)


def test_target_inverse_then_reduction_then_final_output() -> None:
    outputs = torch.tensor([[0.0], [torch.log(torch.tensor(4.0))]])
    model = _AnalyticModel(outputs)
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))
    recipe = Recipe(
        target=[_LogTarget()],
        output=[
            TaskDispatch(
                classification=ClassDecode(),
                regression=TargetDecode(),
            ),
            EstimatorMean(),
            TaskDispatch(
                classification=Identity(),
                regression=_SquareOutput(),
            ),
        ],
    )

    actual = model(
        _features(3),
        target,
        _features(1),
        recipe=recipe,
        num_estimators=2,
    )

    torch.testing.assert_close(actual.numerical, torch.tensor([[6.25]]))
    inverse_after_average = outputs.mean(dim=0).exp().square().unsqueeze(0)
    output_before_average = torch.tensor([1.0, 16.0]).mean().reshape(1, 1)
    assert not torch.equal(actual.numerical, inverse_after_average)
    assert not torch.equal(actual.numerical, output_before_average)


def test_recipe_preserves_estimator_order_through_reduction() -> None:
    model = _AnalyticModel(torch.tensor([[1.0], [3.0]]))
    target = TableTensor.from_tensor(torch.ones(3, 1))
    recipe = Recipe(
        target=[_LogTarget()],
        output=[
            TaskDispatch(
                classification=ClassDecode(),
                regression=TargetDecode(),
            ),
            _OrderedReduction(),
            TaskDispatch(
                classification=Identity(),
                regression=Identity(),
            ),
        ],
    )

    actual = model(
        _features(3),
        target,
        _features(1),
        recipe=recipe,
        num_estimators=2,
    )

    expected = (
        torch.tensor(1.0).exp() + 10 * torch.tensor(3.0).exp()
    ).reshape(1, 1)
    torch.testing.assert_close(actual.numerical, expected)


def test_tabicl_recipe_exposes_one_complete_output_pipeline() -> None:
    output = TabICLv2.default_recipe().output

    assert isinstance(output, Sequential)
    assert len(output.steps) == 3
    assert isinstance(output.steps[0], TaskDispatch)
    assert isinstance(output.steps[1], EstimatorMean)
    assert isinstance(output.steps[2], TaskDispatch)
    first = output.steps[0]
    assert isinstance(first.processors["classification"], ClassDecode)
    assert isinstance(first.processors["regression"], TargetDecode)


def test_rfm_recipe_decodes_quantiles_before_target_inverse() -> None:
    model = _AnalyticModel(
        torch.tensor(
            [
                [2.0, -2.0, 0.0],
                [3.0, -1.0, 1.0],
            ]
        )
    )
    target = TableTensor.from_tensor(torch.tensor([[10.0], [20.0], [30.0]]))
    recipe = KumoRFM.default_recipe()

    first = cast(Sequential, recipe.output).steps[0]
    assert isinstance(first, TaskDispatch)
    regression = first.processors["regression"]
    assert isinstance(regression, Sequential)
    assert isinstance(regression.steps[0], QuantileDecode)
    assert isinstance(regression.steps[1], TargetDecode)

    actual = model(
        _features(3),
        target,
        _features(1),
        recipe=recipe,
        num_estimators=2,
    )

    scale = target.numerical.std(dim=0, correction=0)
    expected = (torch.tensor(0.5) * scale + target.numerical.mean()).reshape(
        1, 1
    )
    torch.testing.assert_close(actual.numerical, expected)
