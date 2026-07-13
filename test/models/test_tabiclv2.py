import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.processing import Sequential
from sdm.testing import withCUDA


def _make_small_classifier(
    max_classes: int,
    device: torch.device,
) -> _TabICLv2:
    return _TabICLv2(
        num_classes=max_classes,
        num_quantiles=0,
        channels=8,
        num_embedding_layers=1,
        num_embedding_heads=2,
        num_inducing_points=4,
        group_size=3,
        num_readout_tokens=2,
        num_icl_layers=1,
        num_icl_heads=2,
        norm_bias=True,
        device=device,
    )


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
        y = torch.randint(0, 10, (*batch_shape, R_train), device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 10)

    assert out.dtype == x.dtype
    assert out.device == x.device

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

    with pytest.raises(ValueError, match="num_estimators"):
        model(x, y, num_estimators=0)

    with pytest.raises(ValueError, match="num_estimators"):
        model.fit(x[..., :R_train, :], y, num_estimators=0)


def test_default_recipe_regression_roundtrip() -> None:
    recipe = TabICLv2(pretrained=False).default_recipe()

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
    assert model_features.columns[Stype.numerical] == (
        "a",
        "b",
        "c",
        "d",
        "kind",
    )

    assert isinstance(recipe.target, Sequential)
    restored = recipe.target.inverse_transform(model_target)
    torch.testing.assert_close(
        restored.numerical, target.numerical, atol=1e-4, rtol=1e-4
    )


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


@withCUDA
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_tabiclv2_many_classes(
    device: torch.device,
    batch_shape: tuple[int, ...],
) -> None:
    model = _make_small_classifier(max_classes=3, device=device)
    num_classes, test_size = 7, 2
    x = torch.randn(
        *batch_shape,
        num_classes + test_size,
        6,
        device=device,
    )
    y = torch.arange(num_classes, device=device)
    y = y.expand(*batch_shape, num_classes)

    out = model(x, y)

    assert out.size() == (*batch_shape, test_size, num_classes)
    assert out.dtype == x.dtype
    assert out.device == device
    assert torch.isfinite(out).all()
    probabilities = (out / 0.9).softmax(dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(*batch_shape, test_size, device=device),
    )

    if len(batch_shape) > 0:
        looped = torch.stack([model(x[i], y[i]) for i in range(x.size(0))])
        torch.testing.assert_close(out, looped)


def test_tabiclv2_hierarchical_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IdentityRowEmbedding(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            *,
            cache: object | None = None,
        ) -> torch.Tensor:
            return x

    class NodePredictor(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
        ) -> torch.Tensor:
            test_size = x.size(0) - y.size(0)
            if y.size(0) == 3:
                probabilities = x.new_tensor([0.6, 0.4])
            else:
                probabilities = x.new_tensor([0.25, 0.75])
            return probabilities.log().mul(0.9).expand(test_size, -1)

    model = _make_small_classifier(
        max_classes=2,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(model, "row_embedding", IdentityRowEmbedding())
    monkeypatch.setattr(model, "icl_block", NodePredictor())
    monkeypatch.setattr(model, "head", torch.nn.Identity())

    y = torch.tensor([[0, 1, 0], [0, 1, 2]])
    out = model(torch.randn(2, 5, 4), y)

    probabilities = torch.tensor([[0.6, 0.4, 0.0], [0.15, 0.45, 0.4]])
    probabilities = probabilities.unsqueeze(1).expand(-1, 2, -1)
    expected = (probabilities + 1e-6).log().mul(0.9)
    torch.testing.assert_close(out, expected)


@withCUDA
def test_tabiclv2_heterogeneous_class_batch(device: torch.device) -> None:
    model = _make_small_classifier(max_classes=3, device=device)
    x = torch.randn(2, 6, 6, device=device)
    y = torch.tensor(
        [[0, 1, 2, 0], [0, 1, 2, 3]],
        device=device,
    )

    out = model(x, y)

    assert out.size() == (2, 2, 4)
    probabilities = (out / 0.9).softmax(dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, 2, device=device),
    )
    assert (probabilities[0, :, 3] < 1e-5).all()


@pytest.mark.parametrize("num_classes", [10, 11])
def test_tabiclv2_native_class_boundary(num_classes: int) -> None:
    model = _make_small_classifier(
        max_classes=10,
        device=torch.device("cpu"),
    )
    test_size = 2
    x = torch.randn(num_classes + test_size, 6)
    y = torch.arange(num_classes)

    out = model(x, y)

    assert out.size() == (test_size, num_classes)


def test_tabiclv2_many_classes_rejects_cache() -> None:
    model = TabICLv2(pretrained=False)
    native_x = torch.randn(5, 6)
    native_y = torch.tensor([0, 1, 2, 0, 1])
    model.fit(native_x, native_y)
    assert model.predict(torch.randn(1, 6)).size(-1) == 10

    x = torch.randn(11, 6)
    y = torch.arange(11)

    with pytest.raises(
        NotImplementedError,
        match="caching is not supported with more than 10 classes",
    ):
        model.fit(x, y)

    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(torch.randn(2, 6))


@withCUDA
def test_tabiclv2_many_classes_forward(device: torch.device) -> None:
    model = TabICLv2(pretrained=False, device=device)
    num_classes, test_size = 11, 2
    x = torch.randn(num_classes + test_size, 6, device=device)
    y = torch.arange(num_classes, dtype=torch.int32, device=device)

    out = model(x, y)

    assert out.size() == (test_size, num_classes)
    assert torch.isfinite(out).all()
    probabilities = (out / 0.9).softmax(dim=-1)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(test_size, device=device),
    )
