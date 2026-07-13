from typing import ClassVar, cast

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
    Identity,
    InvertibleMixin,
    Processor,
    Recipe,
    SoftmaxTemperature,
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


class _AnalyticModel(Model):
    supports_related_tables = False

    def __init__(self, outputs: Tensor) -> None:
        super().__init__()
        self.register_buffer("_outputs_buffer", outputs)
        self._uncached_calls = 0
        self._cache_build_calls = 0

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


def test_classification_maps_members_before_logit_aggregation() -> None:
    _CyclingClassShuffle.next_plan = 0
    canonical, raw = _classification_outputs()

    actual = _AnalyticModel(raw)(
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


def test_regression_target_inverse_runs_before_aggregation() -> None:
    outputs = torch.tensor([[0.0], [torch.log(torch.tensor(4.0))]])
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))

    actual = _AnalyticModel(outputs)(
        _features(),
        target,
        recipe=Recipe(target=[_LogTarget()], output=[Identity()]),
        num_estimators=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[2.5]]))
    assert not torch.equal(
        actual,
        outputs.mean(dim=0).exp().unsqueeze(0),
    )


def test_output_transform_runs_once_after_aggregation() -> None:
    target = TableTensor.from_tensor(torch.ones(3, 1))

    actual = _AnalyticModel(torch.tensor([[1.0], [3.0]]))(
        _features(),
        target,
        recipe=Recipe(target=[Identity()], output=[_SquareOutput()]),
        num_estimators=2,
    )

    torch.testing.assert_close(actual, torch.tensor([[4.0]]))
    assert not torch.equal(actual, torch.tensor([[5.0]]))


def test_cached_classification_reuses_category_mapping() -> None:
    canonical, raw = _classification_outputs()
    _CyclingClassShuffle.next_plan = 1
    model = _AnalyticModel(raw[1:])
    model.fit(
        _features()[:3],
        _classification_target(),
        recipe=_classification_recipe(),
    )

    actual = model.predict(_features()[3:])
    expected = (canonical[1] / 0.9).softmax(dim=-1).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
