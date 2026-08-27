from typing import ClassVar, Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import model as model_module


class _RecordingCore(torch.nn.Module):
    calls: ClassVar[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int | None]]
    ] = []

    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert (num_classes, num_quantiles) in ((10, 0), (0, 999))
        self.num_classes = num_classes
        self.num_quantiles = num_quantiles
        self.anchor = torch.nn.Parameter(
            torch.empty((), device=device, dtype=dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        categorical_mask: torch.Tensor,
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> torch.Tensor:
        type(self).calls.append(
            (
                x.clone(),
                y.clone(),
                categorical_mask.clone(),
                cache is not None and cache.is_replaying,
                batch_size_limit,
            )
        )
        query = x[..., y.size(-1) :, :]
        if self.num_classes > 0:
            offsets = torch.arange(
                self.num_classes,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            offsets = torch.linspace(
                1.0,
                3.0,
                self.num_quantiles,
                device=x.device,
                dtype=x.dtype,
            )
        return query.sum(dim=-1, keepdim=True) + offsets


@pytest.fixture
def recording_model(monkeypatch: pytest.MonkeyPatch) -> KumoTabular:
    _RecordingCore.calls.clear()
    monkeypatch.setattr(model_module, "_KumoTabular", _RecordingCore)
    return KumoTabular()


def _features() -> tuple[TableTensor, TableTensor]:
    context = TableTensor(
        columns={Stype.numerical: ("n0", "n1"), Stype.categorical: ("c",)},
        numerical=torch.tensor(
            [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]]
        ),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [0]]),
            categories=(torch.tensor([10, 20]),),
        ),
    )
    query = TableTensor(
        columns={Stype.numerical: ("n0", "n1"), Stype.categorical: ("c",)},
        numerical=torch.tensor([[103.0, 203.0], [104.0, 204.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]]),
            categories=(torch.tensor([20, 10]),),
        ),
    )
    return context, query


def _target(num_classes: int = 3) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [2], [0]]),
            categories=(torch.arange(num_classes).mul(10),),
        ),
    )


def _recipe() -> sp.Recipe:
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=sp.AlignCategories(sort_by="value"),
            ),
            sp.ToNumerical(),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.Softmax(),
        ],
    )


def test_default_recipe_reduces_and_applies_classification_softmax(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()

    out = recording_model(context.numerical, _target(), query.numerical)

    assert out.size() == (2, 3)
    torch.testing.assert_close(out.numerical.sum(dim=-1), torch.ones(2))
    x, _, categorical_mask, _, _ = _RecordingCore.calls[0]
    assert x.equal(torch.cat((context.numerical, query.numerical), dim=-2))
    assert not categorical_mask.any()


def test_forward_preserves_declared_classes_and_feature_order(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()

    out = recording_model(
        context,
        _target(),
        query,
        recipe=_recipe(),
        batch_size_limit=7,
    )

    assert out.size() == (2, 3)
    assert out.columns[Stype.numerical] == ("0", "10", "20")
    torch.testing.assert_close(out.numerical.sum(dim=-1), torch.ones(2))

    x, y, categorical_mask, replaying, batch_size_limit = _RecordingCore.calls[
        0
    ]
    assert categorical_mask.tolist() == [False, False, True]
    assert x[..., :2].equal(
        torch.cat((context.numerical, query.numerical), dim=-2)
    )
    # Query categories use the reversed input vocabulary and are realigned to
    # the context vocabulary before becoming the final numerical column.
    assert x[..., -1].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert y.tolist() == [0, 2, 0]
    assert not replaying
    assert batch_size_limit == 7


def test_fit_predict_matches_one_shot_and_replays_categorical_mask(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()
    target = _target()

    expected = recording_model(
        context,
        target,
        query,
        recipe=_recipe(),
        batch_size_limit=3,
    )
    recording_model.fit(
        context,
        target,
        recipe=_recipe(),
        batch_size_limit=3,
    )
    actual = recording_model.predict(query)

    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual.columns == expected.columns
    direct_mask = _RecordingCore.calls[0][2]
    fit_mask = _RecordingCore.calls[1][2]
    replay_mask = _RecordingCore.calls[2][2]
    assert direct_mask.equal(fit_mask)
    assert fit_mask.equal(replay_mask)
    assert not _RecordingCore.calls[1][3]
    assert _RecordingCore.calls[2][3]
    assert _RecordingCore.calls[2][4] == 3


def test_categorical_mask_tracks_shuffled_columns(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()
    recipe = _recipe()
    recipe.append_features(sp.ShuffleColumns(method="shift"))

    recording_model(
        context,
        _target(),
        query,
        recipe=recipe,
        num_estimators=3,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(_RecordingCore.calls) == 3
    for x, _, categorical_mask, _, _ in _RecordingCore.calls:
        assert categorical_mask.sum() == 1
        categorical = x[..., categorical_mask]
        numerical = x[..., ~categorical_mask]
        assert categorical.max() <= 1
        assert numerical.min() >= 100


def test_class_limit_uses_declared_vocabulary(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()
    target = _target(num_classes=10)
    out = recording_model(context, target, query, recipe=_recipe())
    assert out.size() == (2, 10)

    with pytest.raises(ValueError, match="only supports up to 10 classes"):
        recording_model(
            context,
            _target(num_classes=11),
            query,
            recipe=_recipe(),
        )


def test_regression_standardizes_target_and_averages_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingCore.calls.clear()
    monkeypatch.setattr(model_module, "_KumoTabular", _RecordingCore)
    model = KumoTabular(task="regression")
    context = TableTensor.from_tensor(torch.zeros(3, 2))
    query = TableTensor.from_tensor(torch.zeros(2, 2))
    target = TableTensor.from_tensor(torch.tensor([[10.0], [20.0], [30.0]]))

    expected = model(context, target, query)
    model.fit(context, target)
    actual = model.predict(query)

    scale = target.numerical.std(dim=-2, correction=0, keepdim=True)
    point = target.numerical.mean(dim=-2, keepdim=True) + 2.0 * scale
    torch.testing.assert_close(expected.numerical, point.expand(2, 1))
    torch.testing.assert_close(actual.numerical, expected.numerical)
    assert actual.columns[Stype.numerical] == ("pred",)

    normalized_target = _RecordingCore.calls[0][1]
    torch.testing.assert_close(
        normalized_target.mean(dim=-1),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        normalized_target.std(dim=-1, correction=0),
        torch.tensor(1.0),
    )

    tiny_target = TableTensor.from_tensor(
        torch.tensor([[0.0], [1e-9], [2e-9]])
    )
    tiny_out = model(context, tiny_target, query)
    torch.testing.assert_close(
        tiny_out.numerical,
        torch.full((2, 1), 2.0 + 1e-9),
    )


@pytest.mark.parametrize(
    ("task", "target", "target_name"),
    [
        ("classification", torch.arange(3.0).unsqueeze(-1), "numerical"),
        ("regression", _target(), "categorical"),
    ],
)
def test_target_must_match_task(
    monkeypatch: pytest.MonkeyPatch,
    task: Literal["classification", "regression"],
    target: torch.Tensor | TableTensor,
    target_name: str,
) -> None:
    monkeypatch.setattr(model_module, "_KumoTabular", _RecordingCore)
    context, query = _features()
    model = KumoTabular(task=task)

    with pytest.raises(ValueError, match=f"received a {target_name} target"):
        model(context, target, query, recipe=_recipe())
