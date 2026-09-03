from typing import Any, Literal, cast

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models.kumo.tabular import KumoTabular


class _RecordingKumoTabular(KumoTabular):
    def __init__(
        self,
        task: Literal["classification", "regression"] = "regression",
    ) -> None:
        ICLModel.__init__(self, task=task)
        self.contexts: list[tuple[TableTensor, TableTensor]] = []
        self.queries: list[TableTensor] = []

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        if x_context is not None:
            assert y_context is not None
            self.contexts.append((x_context, y_context))
        if x_query is None:
            assert x_context is not None
            return x_context

        self.queries.append(x_query)
        if Task.classification not in self.tasks:
            return x_query

        if y_context is not None:
            classes = y_context.categorical.categories[0]
        else:
            assert cache is not None
            classes = cast(torch.Tensor, cache["classes"])
        return TableTensor(
            columns={
                Stype.numerical: tuple(
                    str(value) for value in classes.tolist()
                )
            },
            numerical=torch.zeros(
                (x_query.size(-2), len(classes)),
                device=x_query.device,
                dtype=x_query.dtype,
            ),
        )


def _build(task: Literal["classification", "regression"]) -> KumoTabular:
    model = KumoTabular(task=task, pretrained=False)
    # Residual branches are zero-initialized, so an untrained model maps every
    # row onto the same constant. Randomize them to make the prediction depend
    # on the features it is given.
    for parameter in model.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    return model


@pytest.fixture
def cls_model() -> KumoTabular:
    return _build("classification")


@pytest.fixture
def reg_model() -> KumoTabular:
    return _build("regression")


def _features(stype: Stype = Stype.categorical) -> tuple[TableTensor, ...]:
    numerical = torch.tensor(
        [
            [100.0, 200.0],
            [101.0, 201.0],
            [102.0, 202.0],
            [103.0, 203.0],
            [104.0, 204.0],
        ]
    )
    label = torch.tensor([[0], [1], [0], [0], [1]])
    if stype == Stype.categorical:
        x = TableTensor(
            columns={Stype.numerical: ("n0", "n1"), Stype.categorical: ("c",)},
            numerical=numerical,
            categorical=CategoricalTensor.from_tensor(label),
        )
    else:
        x = TableTensor(
            columns={Stype.numerical: ("n0", "n1", "c")},
            numerical=torch.cat((numerical, label.float()), dim=-1),
        )
    return x.split(3, dim=0)


def _cls_target() -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [2], [0]]),
            categories=(torch.arange(3).mul(10),),
        ),
    )


def _reg_target(offset: float = 0.0) -> TableTensor:
    values = torch.tensor([[10.0], [20.0], [30.0]])
    return TableTensor.from_tensor(values + offset)


def _recipe() -> sp.Recipe:
    return sp.Recipe(
        features=[sp.ToNumerical()],
        target=sp.StypeDispatch(numerical=sp.Standardize()),
        output=[sp.ReduceEstimators(method="mean")],
    )


def test_forward(
    cls_model: KumoTabular,
    reg_model: KumoTabular,
) -> None:
    x_context, x_query = _features()

    out = cls_model(x_context, _cls_target(), x_query, recipe=_recipe())

    assert out.size() == (2, 3)
    assert out.columns[Stype.numerical] == ("0", "10", "20")
    assert out.dtype == x_query.dtype
    assert torch.is_inference(out)

    x_context, x_query = _features(Stype.numerical)
    out = reg_model(
        x_context,
        _reg_target(),
        x_query,
        recipe=_recipe(),
    )

    assert out.size() == (2, 999)
    assert out.columns[Stype.numerical] == tuple(
        f"q{i:03d}" for i in range(1, 1000)
    )


def test_categorical_features_are_marked(cls_model: KumoTabular) -> None:
    target = _cls_target()

    x_context, x_query = _features(Stype.categorical)
    categorical = cls_model(x_context, target, x_query, recipe=_recipe())
    # Declaring the same column numerical leaves the features handed to the
    # model untouched, so only the stype it embeds them with differs.
    x_context, x_query = _features(Stype.numerical)
    numerical = cls_model(x_context, target, x_query, recipe=_recipe())

    assert not categorical.allclose(numerical)


def test_fit_predict(
    cls_model: KumoTabular,
    reg_model: KumoTabular,
) -> None:
    x_context, x_query = _features()
    target = _cls_target()

    expected = cls_model(x_context, target, x_query, recipe=_recipe())
    cls_model.fit(x_context, target, recipe=_recipe())
    actual = cls_model.predict(x_query)

    assert actual.allclose(expected, atol=1e-5)
    assert actual.columns == expected.columns

    x_context, x_query = _features(Stype.numerical)
    target = _reg_target()
    recipe = _recipe()
    expected = reg_model(x_context, target, x_query, recipe=recipe)
    reg_model.fit(x_context, target, recipe=recipe)
    actual = reg_model.predict(x_query)

    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=1e-4,
        rtol=1e-4,
    )


def test_fit_partitions_shared_context_across_estimators() -> None:
    model = _RecordingKumoTabular()
    x = torch.arange(11, dtype=torch.float).unsqueeze(-1)
    y = x + 100

    model.fit(
        x,
        y,
        recipe=sp.Recipe(),
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )

    assert [context_x.size(-2) for context_x, _ in model.contexts] == [4, 4, 3]
    selected_x = torch.cat(
        [context_x.numerical for context_x, _ in model.contexts]
    )
    selected_y = torch.cat(
        [context_y.numerical for _, context_y in model.contexts]
    )
    assert selected_y.equal(selected_x + 100)
    assert selected_x.sort(dim=-2).values.equal(x)


def test_fit_partitions_after_target_preprocessing() -> None:
    model = _RecordingKumoTabular()
    x = torch.arange(11, dtype=torch.float).unsqueeze(-1)
    y = x + 100
    expected_y = (
        sp.Standardize().fit_transform(TableTensor.from_tensor(y)).numerical
    )

    model.fit(
        x,
        y,
        recipe=sp.Recipe(target=sp.Standardize()),
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )

    for context_x, context_y in model.contexts:
        row_ids = context_x.numerical.squeeze(-1).long()
        assert context_y.numerical.equal(expected_y[row_ids])


def test_fit_partition_is_deterministic_with_generator() -> None:
    x = torch.arange(11, dtype=torch.float).unsqueeze(-1)
    y = x + 100

    contexts = []
    for _ in range(2):
        model = _RecordingKumoTabular()
        model.fit(
            x,
            y,
            recipe=sp.Recipe(),
            num_estimators=3,
            generator=torch.Generator().manual_seed(0),
        )
        contexts.append(
            tuple(context_x.numerical for context_x, _ in model.contexts)
        )

    assert all(
        first.equal(second) for first, second in zip(*contexts, strict=True)
    )


def test_partitioned_fit_predict_matches_forward() -> None:
    x_context = torch.arange(9, dtype=torch.float).unsqueeze(-1)
    y_context = x_context + 100
    x_query = torch.arange(5, dtype=torch.float).unsqueeze(-1) + 20
    recipe = sp.Recipe(output=sp.ReduceEstimators())

    direct_model = _RecordingKumoTabular()
    expected = direct_model(
        x_context,
        y_context,
        x_query,
        recipe=recipe,
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )

    cached_model = _RecordingKumoTabular()
    cached_model.fit(
        x_context,
        y_context,
        recipe=recipe,
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )
    actual = cached_model.predict(x_query)

    assert actual.equal(expected)
    assert all(
        query.numerical.equal(x_query) for query in direct_model.queries
    )
    assert all(
        query.numerical.equal(x_query) for query in cached_model.queries
    )
    for direct, cached in zip(
        direct_model.contexts,
        cached_model.contexts,
        strict=True,
    ):
        assert direct[0].equal(cached[0])
        assert direct[1].equal(cached[1])


def test_partitioned_query_uses_member_feature_state() -> None:
    model = _RecordingKumoTabular()
    x_context = torch.arange(12, dtype=torch.float).unsqueeze(-1)
    y_context = x_context + 100
    x_query = torch.tensor([[20.0], [30.0]])

    model.fit(
        x_context,
        y_context,
        recipe=sp.Recipe(features=sp.Standardize()),
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )
    model.predict(x_query)

    permutation = torch.randperm(
        x_context.size(0),
        generator=torch.Generator().manual_seed(0),
    )
    for member_id, rows in enumerate(permutation.view(3, 4)):
        standardizer = sp.Standardize()
        standardizer.fit(TableTensor.from_tensor(x_context[rows]))
        expected = standardizer.transform(TableTensor.from_tensor(x_query))
        torch.testing.assert_close(
            model.queries[member_id].numerical,
            expected.numerical,
        )


def test_partition_preserves_global_target_classes() -> None:
    model = _RecordingKumoTabular(task="classification")
    x = torch.arange(7, dtype=torch.float).unsqueeze(-1)
    y = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [0], [0], [0], [0], [0], [1]]),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    recipe = sp.Recipe(
        target=sp.AlignCategories(),
        output=sp.ReduceEstimators(),
    )

    model.fit(
        x,
        y,
        recipe=recipe,
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )
    prediction = model.predict(torch.tensor([[100.0]]))

    assert prediction.columns[Stype.numerical] == ("10", "20")
    assert any(
        not context_y.categorical.code.eq(1).any()
        for _, context_y in model.contexts
    )
    assert all(
        context_y.categorical.categories[0].equal(torch.tensor([10, 20]))
        for _, context_y in model.contexts
    )


def test_fit_keeps_shared_context_for_one_estimator() -> None:
    model = _RecordingKumoTabular()
    x = torch.arange(5, dtype=torch.float).unsqueeze(-1)
    y = x + 100

    model.fit(
        x,
        y,
        recipe=sp.Recipe(),
        num_estimators=1,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(model.contexts) == 1
    assert model.contexts[0][0].numerical.equal(x)
    assert model.contexts[0][1].numerical.equal(y)


def test_fit_keeps_shared_context_when_estimators_outnumber_rows() -> None:
    model = _RecordingKumoTabular()
    x = torch.arange(2, dtype=torch.float).unsqueeze(-1)
    y = x + 100

    model.fit(
        x,
        y,
        recipe=sp.Recipe(),
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(model.contexts) == 3
    for context_x, context_y in model.contexts:
        assert context_x.numerical.equal(x)
        assert context_y.numerical.equal(y)


def test_fit_rejects_mismatched_rows_before_partitioning() -> None:
    model = _RecordingKumoTabular()

    with pytest.raises(ValueError, match="matching row dimensions"):
        model.fit(
            torch.randn(5, 2),
            torch.randn(4, 1),
            recipe=sp.Recipe(),
            num_estimators=2,
        )


def test_fit_respects_leading_estimator_dimension() -> None:
    model = _RecordingKumoTabular()
    x = torch.arange(8, dtype=torch.float).view(2, 4, 1)
    y = x + 100

    model.fit(x, y, recipe=sp.Recipe())

    assert len(model.contexts) == 2
    for member_id, (context_x, context_y) in enumerate(model.contexts):
        assert context_x.numerical.equal(x[member_id])
        assert context_y.numerical.equal(y[member_id])


def test_default_recipe_fit_predict_multiple_estimators() -> None:
    model = KumoTabular(task="regression", size="small", pretrained=False)
    x_context = torch.randn(12, 3)
    y_context = torch.randn(12, 1)
    x_query = torch.randn(2, 3)

    model.fit(x_context, y_context, num_estimators=2)
    out = model.predict(x_query)

    assert out.size() == (2, 999)
