from pathlib import Path
from typing import ClassVar, Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, NaT, StringTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import model as model_module
from sdm.models.kumo.tabular.model import _KumoTabular


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


def test_parameter_count() -> None:
    model = _KumoTabular(
        num_classes=10,
        num_quantiles=0,
        device="meta",
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        34_188_428
    )


def test_checkpoint_path_uses_meta_core_and_requested_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingCore(torch.nn.Module):
        def __init__(
            self,
            num_classes: int,
            num_quantiles: int,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.num_quantiles = num_quantiles
            self.anchor = torch.nn.Parameter(
                torch.empty((), device=device, dtype=dtype)
            )

    calls: list[tuple[Path, str, torch.device | str | None]] = []
    monkeypatch.setattr(model_module, "_KumoTabular", RecordingCore)

    def fake_load_checkpoint(
        model: RecordingCore,
        checkpoint_path: str | Path,
        *,
        task: Literal["classification", "regression"],
        device: torch.device | str | None,
    ) -> RecordingCore:
        assert model.anchor.device.type == "meta"
        calls.append((Path(checkpoint_path), task, device))
        return RecordingCore(
            num_classes=model.num_classes,
            num_quantiles=model.num_quantiles,
            device=device,
        ).eval()

    monkeypatch.setattr(model_module, "load_checkpoint", fake_load_checkpoint)
    path = tmp_path / "regression.pt"

    model = KumoTabular(
        "cpu",
        task="regression",
        checkpoint_path=path,
    )

    assert calls == [(path, "regression", "cpu")]
    assert model.model.anchor.device.type == "cpu"
    assert not model.training
    assert not model.model.training


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


def _build(task: Literal["classification", "regression"]) -> KumoTabular:
    model = KumoTabular(task=task)
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


def _mixed_features() -> tuple[TableTensor, TableTensor]:
    context = TableTensor(
        columns={
            Stype.numerical: ("num",),
            Stype.datetime: ("when",),
            Stype.categorical: ("cat",),
        },
        numerical=torch.tensor([[1.0], [torch.inf], [-torch.inf]]),
        datetime=torch.tensor([[0], [NaT], [2 * 86_400_000_000]]),
        categorical=CategoricalTensor(
            code=torch.full((3, 1), -1),
            categories=(StringTensor.from_list(["unused"]),),
        ),
    )
    query = TableTensor(
        columns={
            Stype.numerical: ("num",),
            Stype.datetime: ("when",),
            Stype.categorical: ("cat",),
        },
        numerical=torch.tensor([[5.0], [torch.nan]]),
        datetime=torch.tensor([[3 * 86_400_000_000], [NaT]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [-1]]),
            categories=(StringTensor.from_list(["unseen"]),),
        ),
    )
    return context, query


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


def test_default_recipe_reduces_and_applies_classification_softmax(
    recording_model: KumoTabular,
) -> None:
    context, query = _features()

    out = recording_model(context.numerical, _cls_target(), query.numerical)

    assert out.size() == (2, 3)
    torch.testing.assert_close(out.numerical.sum(dim=-1), torch.ones(2))
    x, _, categorical_mask, _, _ = _RecordingCore.calls[0]
    torch.testing.assert_close(
        x[:3],
        torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
    )
    assert not categorical_mask.any()


def test_default_recipe_handles_missing_values_and_preserves_feature_roles(
    recording_model: KumoTabular,
) -> None:
    context, query = _mixed_features()
    features = KumoTabular.default_recipe().features

    transformed = features.fit_transform(context)
    assert transformed.columns[Stype.numerical] == (
        "num",
        "when__month__sin",
        "when__month__cos",
        "when__day_of_month__sin",
        "when__day_of_month__cos",
        "when__hour__sin",
        "when__hour__cos",
        "when__weekday__sin",
        "when__weekday__cos",
        "cat",
    )

    recording_model(context, _cls_target(), query)

    x, _, categorical_mask, _, _ = _RecordingCore.calls[0]
    assert x.isfinite().all()
    assert categorical_mask.tolist() == [False] * 9 + [True]
    # An all-missing context has no fitted mode, so both context and query
    # retain the finite alignment sentinel through standardization.
    assert x[..., -1].equal(torch.zeros(5))


def test_default_recipe_distinguishes_missing_and_unseen_categories() -> None:
    context = TableTensor.from_columns(
        {"cat": ["b", None, "b", "a"]},
        {"cat": "categorical"},
        categorical_as_string=True,
        categorical_missing_value="___missing___",
    )
    query = TableTensor.from_columns(
        {"cat": ["unseen", None]},
        {"cat": "categorical"},
        categorical_as_string=True,
        categorical_missing_value="___missing___",
    )
    features = KumoTabular.default_recipe().features

    transformed_context = features.fit_transform(context).numerical
    transformed_query = features.transform(query).numerical

    assert transformed_query[0].equal(transformed_context[0])
    assert transformed_query[1].equal(transformed_context[1])


def test_default_recipe_clips_categories_but_not_datetime_fields() -> None:
    num_rows = 100
    hours = torch.zeros(num_rows, dtype=torch.long)
    hours[-1] = 12
    categories = torch.arange(2)
    codes = torch.zeros(num_rows, 1, dtype=torch.long)
    codes[-1] = 1
    context = TableTensor(
        columns={Stype.datetime: ("when",), Stype.categorical: ("cat",)},
        datetime=hours.mul(3_600_000_000).unsqueeze(-1),
        categorical=CategoricalTensor(
            code=codes,
            categories=(categories,),
        ),
    )

    transformed = KumoTabular.default_recipe().features.fit_transform(context)

    hour_cos = transformed.columns[Stype.numerical].index("when__hour__cos")
    cat = transformed.columns[Stype.numerical].index("cat")
    assert transformed.numerical[-1, hour_cos] < -4
    assert transformed.numerical[-1, cat] < 4.1


def test_default_recipe_fits_numerical_state_only_on_context() -> None:
    context = TableTensor.from_tensor(torch.arange(20.0).unsqueeze(-1))
    fixed_query = torch.tensor([[10.5]])
    query_a = TableTensor.from_tensor(
        torch.cat((fixed_query, torch.tensor([[1e20]])))
    )
    query_b = TableTensor.from_tensor(
        torch.cat((fixed_query, torch.tensor([[-1e20]])))
    )
    features = KumoTabular.default_recipe().features.fit(context)

    out_a = features.transform(query_a).numerical
    out_b = features.transform(query_b).numerical

    assert out_a.isfinite().all()
    assert out_b.isfinite().all()
    assert out_a[0].equal(out_b[0])
    assert not out_a[1].equal(out_b[1])


def test_default_recipe_preserves_precision_before_model_cast(
    recording_model: KumoTabular,
) -> None:
    context = TableTensor.from_columns(
        {"value": (1e12 + torch.arange(16, dtype=torch.float64)).tolist()},
        {"value": "numerical"},
        numerical_dtype=torch.float64,
    )
    query = TableTensor.from_columns(
        {"value": [1e12 + 16]},
        {"value": "numerical"},
        numerical_dtype=torch.float64,
    )
    target = TableTensor(
        categorical=CategoricalTensor(
            code=torch.arange(16).remainder(3).unsqueeze(-1),
            categories=(torch.arange(3).mul(10),),
        )
    )

    recording_model(context, target, query)

    x, _, _, _, _ = _RecordingCore.calls[0]
    assert x.dtype == torch.float32
    assert x[:16].unique().numel() == 16


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
