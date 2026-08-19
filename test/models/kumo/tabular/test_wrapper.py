from pathlib import Path
from typing import ClassVar

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import wrapper as wrapper_module


class _RecordingCore(torch.nn.Module):
    calls: ClassVar[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int | None]]
    ] = []

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
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
        offsets = torch.arange(10, device=x.device, dtype=x.dtype)
        return query.sum(dim=-1, keepdim=True) + offsets


@pytest.fixture
def recording_model(monkeypatch: pytest.MonkeyPatch) -> KumoTabular:
    _RecordingCore.calls.clear()
    monkeypatch.setattr(wrapper_module, "_KumoTabular", _RecordingCore)
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


def test_forward_preserves_declared_classes_and_feature_order(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()

    out = recording_model(
        context,
        _target(),
        query,
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

    expected = recording_model(context, target, query, batch_size_limit=3)
    recording_model.fit(context, target, batch_size_limit=3)
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
    recipe = KumoTabular.default_recipe()
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
    out = recording_model(context, target, query)
    assert out.size() == (2, 10)

    with pytest.raises(ValueError, match="only supports up to 10 classes"):
        recording_model(context, _target(num_classes=11), query)


def test_checkpoint_loader_constructs_public_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _RecordingCore()
    calls: list[
        tuple[torch.nn.Module, str | Path, torch.device | str | None]
    ] = []

    def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str | Path,
        device: torch.device | str | None = None,
    ) -> _RecordingCore:
        calls.append((model, checkpoint_path, device))
        return loaded

    monkeypatch.setattr(wrapper_module, "load_checkpoint", load_checkpoint)
    model = KumoTabular(checkpoint_path="checkpoint.pt", device="cpu")

    assert model.model is loaded
    assert len(calls) == 1
    allocated, checkpoint_path, device = calls[0]
    assert next(allocated.parameters()).is_meta
    assert checkpoint_path == "checkpoint.pt"
    assert device == "cpu"
    assert not model.training
