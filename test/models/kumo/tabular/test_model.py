from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import KumoTabular
from sdm.models.kumo.tabular import model as model_module
from sdm.models.kumo.tabular.model import _KumoTabular


@pytest.mark.parametrize(
    ("num_classes", "num_quantiles", "expected"),
    [
        (10, 0, 27_441_726),
        (0, 999, 28_449_051),
    ],
)
def test_parameter_count(
    num_classes: int,
    num_quantiles: int,
    expected: int,
) -> None:
    model = _KumoTabular(
        num_classes=num_classes,
        num_quantiles=num_quantiles,
        device="meta",
    )

    assert (
        sum(parameter.numel() for parameter in model.parameters()) == expected
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
