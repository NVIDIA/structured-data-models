from typing import cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.processing import (
    Choice,
    HardClip,
    InvertibleMixin,
    Power,
    Recipe,
    SoftmaxTemperature,
)
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_tabiclv2(
    device: torch.device,
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "TabICLv2()"
    else:
        assert repr(model) == "TabICLv2(device=cuda:0)"

    R, C, R_train = 8, 6, 5

    x = torch.randn(*batch_shape, R, C, device=device)
    if dtype.is_floating_point:
        y = torch.randn(*batch_shape, R_train, device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 999)
    else:
        # TODO Increase max value once TabICLv2 supports 10+ classes:
        y = torch.randint(0, 10, (*batch_shape, R_train), device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 10)

    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.is_inference(out)

    if len(batch_shape) > 0:
        looped = torch.stack(
            [model(x[i], y[i]) for i in range(batch_shape[0])]
        )
        torch.testing.assert_close(out, looped)

    model.fit(x[..., :R_train, :], y)
    torch.testing.assert_close(model.predict(x[..., R_train:, :]), out)
    model.clear()


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_tabiclv2_num_estimators(batch_shape: tuple[int, ...]) -> None:
    model = TabICLv2(pretrained=False)

    R, C, R_train = 8, 6, 5
    x = torch.randn(*batch_shape, R, C)
    y = torch.randint(0, 10, (*batch_shape, R_train))

    out = model(x, y)

    # Members are identical for now, so their average matches a single member:
    ensembled = model(x, y, num_estimators=3)
    assert ensembled.size() == out.size()
    torch.testing.assert_close(ensembled, out)

    model.fit(x[..., :R_train, :], y, num_estimators=3)
    torch.testing.assert_close(model.predict(x[..., R_train:, :]), out)
    model.clear()


def test_tabiclv2_recipe() -> None:
    model = TabICLv2(pretrained=False)

    R, C, R_train = 8, 6, 5
    x = TableTensor.from_tensor(torch.randn(R, C))
    y = TableTensor.from_tensor(torch.randn(R_train, 1))

    # An empty recipe matches the recipe-less forward pass:
    torch.testing.assert_close(
        model(x, y, recipe=Recipe()),
        model(x, y),
    )

    # Output steps run after the model and ensembling:
    raw = model(x, y)
    output_recipe = Recipe(output=[SoftmaxTemperature()])
    out = model(x, y, recipe=output_recipe)
    torch.testing.assert_close(out, raw.softmax(dim=-1))
    model.fit(x[:R_train], y, recipe=output_recipe)
    torch.testing.assert_close(model.predict(x[R_train:]), out)

    # The recipe matches its manual driver-side application:
    out = model(x, y, recipe=model.default_recipe())
    assert out.size() == (R - R_train, 999)
    assert torch.is_inference(out)
    recipe = model.default_recipe()
    recipe.features.fit(x[:R_train])
    raw = model(
        x=recipe.features.transform(x),
        y=recipe.target.fit_transform(y),
    )
    expected = (
        cast(InvertibleMixin, recipe.target)
        .inverse_transform(TableTensor.from_tensor(raw.clone()))
        .numerical
    )
    torch.testing.assert_close(out, expected)

    # The fitted recipe state is reused across predict calls:
    model.fit(x[:R_train], y, recipe=model.default_recipe())
    torch.testing.assert_close(model.predict(x[R_train:]), out)
    model.clear()
    assert model._recipes is None


def test_default_recipe_regression_roundtrip() -> None:
    recipe = TabICLv2.default_recipe()

    features = TableTensor(
        columns={
            "numerical": ("a", "b", "c", "d"),
            "categorical": ("kind",),
        },
        numerical=torch.randn(16, 4),
        categorical=CategoricalTensor(
            data=(torch.arange(16, dtype=torch.int32) % 2).unsqueeze(-1),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )
    target = TableTensor.from_tensor(torch.randn(16, 1), columns=["y"])

    model_features = recipe.features.fit_transform(features)
    model_target = recipe.target.fit_transform(target)

    assert model_features.size() == features.size()
    assert model_target.size() == target.size()
    assert model_features.categorical.size(-1) == 0
    assert set(model_features.columns[Stype.numerical]) == {
        "a",
        "b",
        "c",
        "d",
        "kind",
    }

    restored = cast(InvertibleMixin, recipe.target).inverse_transform(
        model_target
    )
    torch.testing.assert_close(
        restored.numerical, target.numerical, atol=1e-4, rtol=1e-4
    )

    assert any(
        isinstance(module, HardClip) for module in recipe.features.modules()
    )
    choice = next(
        module
        for module in recipe.features.modules()
        if isinstance(module, Choice)
    )
    assert any(isinstance(option, Power) for option in choice.options)


def test_default_recipe_binary_classification_smoke() -> None:
    model = TabICLv2(pretrained=False)
    features = TableTensor.from_tensor(
        torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [2.0, -1.0],
                [-1.0, 2.0],
                [0.25, 0.75],
                [1.5, -0.5],
            ]
        ),
        columns=("a", "b"),
    )
    target = TableTensor(
        columns={"categorical": ("label",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [0], [1], [0]]),
            categories=(StringTensor.from_list(["positive", "negative"]),),
        ),
    )

    torch.manual_seed(0)
    direct = model(features, target, recipe=model.default_recipe())

    assert direct.size() == (2, 2)
    assert direct.isfinite().all()
    torch.testing.assert_close(
        direct.sum(dim=-1),
        torch.ones(2),
        rtol=0,
        atol=1e-6,
    )

    torch.manual_seed(0)
    model.fit(features[:5], target, recipe=model.default_recipe())
    cached = model.predict(features[5:])
    torch.testing.assert_close(cached, direct, rtol=1e-5, atol=1e-6)


@withCUDA
def test_row_embedding_mixed_radix_digit(device: torch.device) -> None:
    row_embedding = RowEmbedding(
        num_classes=10,
        channels=8,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
        device=device,
    )
    for module in row_embedding.modules():
        # Randomly initialize to return non-zero output
        if isinstance(module, Attention):
            torch.nn.init.normal_(module.out_lin.weight, std=0.02)

    x = torch.randn(8, 6, device=device)

    # The labels 5 * a + b and 5 * b + a decompose into the digits (a, b) and
    # (b, a) under bases [5, 5], so averaging over digits must be invariant
    # to swapping them:
    a = torch.tensor([4, 0, 1, 2, 3], device=device)
    b = torch.tensor([4, 1, 2, 3, 0], device=device)
    y = 5 * a + b
    y_swapped = 5 * b + a
    out = row_embedding(x, y)
    torch.testing.assert_close(out, row_embedding(x, y_swapped))


@onlyCUDA
@onlyFullTest
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_tabiclv2_compile(dtype: torch.dtype) -> None:
    torch._dynamo.reset()
    model = TabICLv2(pretrained=False, device="cuda")

    R, C, R_train = 8, 6, 5
    x = torch.randn(R, C, device="cuda")
    if dtype.is_floating_point:
        y = torch.randn(R_train, device="cuda")
    else:
        y = torch.randint(0, 10, (R_train,), device="cuda")

    expected = model(x, y)
    submodel = model.reg_model if dtype.is_floating_point else model.cls_model
    submodel.compile(fullgraph=True)

    actual = model(x, y)
    torch.testing.assert_close(actual, expected)
    assert torch.is_inference(actual)

    model.fit(x[:R_train], y)
    predicted = model.predict(x[R_train:])
    torch.testing.assert_close(predicted, expected)
    assert torch.is_inference(predicted)


def test_row_embedding() -> None:
    row_embedding = RowEmbedding(
        num_classes=2,
        channels=8,
        num_layers=1,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=True,
    )

    out = row_embedding(
        x=torch.randn(6, 4),
        y=torch.tensor([0, 1]),
        train_mask=torch.tensor([False, True, False, False, True, False]),
        max_keys=1,
    )
    assert out.size() == (6, 16)
