from typing import Literal

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models.kumo.tabular import KumoTabular


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


def test_numerical_precision_is_preserved_before_model_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingCore(torch.nn.Module):
        seen: torch.Tensor

        def __init__(
            self,
            num_classes: int,
            num_quantiles: int,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.num_classes = num_classes
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
            type(self).seen = x
            return x.new_zeros(
                (*x.size()[:-2], x.size(-2) - y.size(-1), self.num_classes)
            )

    monkeypatch.setattr(model_module, "_KumoTabular", RecordingCore)
    model = KumoTabular()
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

    model(
        context,
        target,
        query,
        recipe=sp.Recipe(features=sp.Standardize()),
    )

    assert RecordingCore.seen.dtype == torch.float32
    assert RecordingCore.seen[:16].unique().numel() == 16


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
