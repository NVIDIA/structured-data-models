from typing import ClassVar

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import model as model_module
from sdm.models.kumo.tabular.model import _KumoTabular


def test_parameter_count() -> None:
    model = _KumoTabular(
        num_classes=10,
        num_quantiles=0,
        device="meta",
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        34_188_428
    )


@pytest.mark.parametrize(
    ("num_classes", "num_quantiles"),
    [(10, 0), (0, 999)],
)
def test_forward_does_not_mutate_input(
    monkeypatch: pytest.MonkeyPatch,
    num_classes: int,
    num_quantiles: int,
) -> None:
    class CellEmbedding(torch.nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(
            self,
            x: torch.Tensor,
            categorical_mask: torch.Tensor,
        ) -> torch.Tensor:
            assert categorical_mask.dtype == torch.bool
            return x.unsqueeze(-1).expand(*x.shape, 128)

    class TableEncoder(torch.nn.Module):
        seen: torch.Tensor

        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(
            self,
            x: torch.Tensor,
            num_context_rows: int,
            *,
            cache: Cache | None,
        ) -> torch.Tensor:
            assert num_context_rows == 2
            assert cache is sentinel_cache
            type(self).seen = x.detach().clone()
            return x.mean(dim=-2).repeat(1, 1, 4)

    class ICLBlock(torch.nn.Module):
        def __init__(self, out_channels: int, **_: object) -> None:
            super().__init__()
            self.out_channels = out_channels

        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            cache: Cache | None,
            batch_size_limit: int | None,
        ) -> torch.Tensor:
            assert cache is sentinel_cache
            assert batch_size_limit == 7
            return x.new_empty(
                (
                    *x.shape[:-2],
                    x.size(-2) - y.size(-1),
                    self.out_channels,
                )
            )

    monkeypatch.setattr(model_module, "CellEmbedding", CellEmbedding)
    monkeypatch.setattr(model_module, "TableEncoder", TableEncoder)
    monkeypatch.setattr(model_module, "ICLBlock", ICLBlock)
    model = _KumoTabular(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
    )

    x = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    original = x.clone()
    y = (
        torch.tensor([[0, 1]])
        if num_classes > 0
        else torch.tensor([[0.25, -1.5]])
    )
    categorical_mask = torch.tensor([[False, True, False, True, False, False]])
    sentinel_cache = Cache()
    out = model(
        x=x,
        y=y,
        categorical_mask=categorical_mask,
        cache=sentinel_cache,
        batch_size_limit=7,
    )

    torch.testing.assert_close(x, original)
    assert out.size() == (1, 2, num_classes or num_quantiles)
    torch.testing.assert_close(
        TableEncoder.seen[..., 2:, :, :],
        original[..., 2:, :, None].expand(1, 2, 6, 128),
    )
    expected_context = original[..., :2, :, None].expand(1, 2, 6, 128)
    y_emb = (
        torch.nn.functional.one_hot(y, num_classes=num_classes)
        if num_classes > 0
        else y.unsqueeze(-1)
    )
    expected_context = expected_context + model.y_encoder(
        y_emb.to(model.y_encoder.weight.dtype)
    ).unsqueeze(-2)
    torch.testing.assert_close(
        TableEncoder.seen[..., :2, :, :],
        expected_context,
    )


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
        assert num_classes == 10
        assert num_quantiles == 0
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


def test_default_recipe_is_empty(recording_model: KumoTabular) -> None:
    context, query = _features()

    out = recording_model(context.numerical, _target(), query.numerical)

    assert out.size() == (1, 2, 3)
    torch.testing.assert_close(
        out.numerical,
        torch.tensor([[[306.0, 307.0, 308.0], [308.0, 309.0, 310.0]]]),
    )
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
