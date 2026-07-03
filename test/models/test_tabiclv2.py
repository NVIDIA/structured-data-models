from typing import cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2.icl import (
    _balanced_grouping,
    _predict_hierarchical,
)
from sdm.models.tabiclv2.model import (
    _class_permutations,
    _probabilities_to_logits,
    _TabICLv2,
    _validate_classification_labels,
)
from sdm.models.tabiclv2.row_embedding import RowEmbedding, _mixed_radix_bases
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
        num_classes = 3
        y = torch.arange(R_train, device=device) % num_classes
        y = y.expand(*batch_shape, R_train)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, num_classes)

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


def test_hierarchical_balanced_grouping() -> None:
    assignments, num_groups = _balanced_grouping(
        num_classes=5,
        max_classes=10,
        device=torch.device("cpu"),
    )
    assert num_groups == 1
    torch.testing.assert_close(assignments, torch.zeros(5, dtype=torch.long))

    assignments, num_groups = _balanced_grouping(
        num_classes=25,
        max_classes=10,
        device=torch.device("cpu"),
    )
    assert num_groups == 3
    torch.testing.assert_close(
        assignments,
        torch.tensor([0] * 9 + [1] * 8 + [2] * 8),
    )

    assignments, num_groups = _balanced_grouping(
        num_classes=101,
        max_classes=10,
        device=torch.device("cpu"),
    )
    assert num_groups == 10
    torch.testing.assert_close(
        assignments,
        torch.arange(10).repeat_interleave(torch.tensor([11] + [10] * 9)),
    )


@withCUDA
@pytest.mark.parametrize(
    ("num_classes", "max_classes", "expected_calls"),
    [(3, 2, 2), (25, 10, 4), (101, 10, 13)],
)
def test_hierarchical_probabilities(
    device: torch.device,
    num_classes: int,
    max_classes: int,
    expected_calls: int,
) -> None:
    calls = 0

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, max_classes))

    test_size = 2
    row_embeddings = torch.randn(
        num_classes + test_size,
        4,
        device=device,
    )
    y = torch.arange(num_classes, device=device)
    probabilities = _predict_hierarchical(
        row_embeddings=row_embeddings,
        y=y,
        num_classes=num_classes,
        max_classes=max_classes,
        temperature=0.9,
        predictor=predictor,
    )

    assert probabilities.size() == (test_size, num_classes)
    assert probabilities.dtype == row_embeddings.dtype
    assert probabilities.device == device
    assert calls == expected_calls
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(test_size, device=device),
    )

    expected = torch.full(
        (num_classes,),
        1 / num_classes,
        device=device,
    )
    if (num_classes, max_classes) == (3, 2):
        expected = torch.tensor([0.25, 0.25, 0.5], device=device)
    elif num_classes == 25:
        expected[:9] = 1 / 27
        expected[9:] = 1 / 24
    elif num_classes == 101:
        expected[:6] = 1 / 120
        expected[6:] = 1 / 100
    torch.testing.assert_close(probabilities[0], expected)


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


def test_hierarchical_probability_to_logit_conversion() -> None:
    probabilities = torch.tensor([[0.2, 0.3, 0.5]])

    logits = _probabilities_to_logits(probabilities)

    expected = 0.9 * (probabilities + 1e-6).log()
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(
        (logits / 0.9).softmax(dim=-1),
        (probabilities + 1e-6) / (probabilities + 1e-6).sum(dim=-1),
    )


@pytest.mark.parametrize(
    "y",
    [
        torch.tensor([-1, 0, 1]),
        torch.tensor([0, 2, 0]),
        torch.tensor([1, 2, 1]),
    ],
)
def test_validate_classification_labels(y: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="contiguous class indices"):
        _validate_classification_labels(y)


def test_validate_classification_labels_rejects_empty_context() -> None:
    with pytest.raises(ValueError, match="at least one in-context"):
        _validate_classification_labels(torch.empty(0, dtype=torch.long))


def test_validate_classification_labels_rejects_mismatched_batch() -> None:
    y = torch.tensor([[0, 1, 0], [0, 1, 2]])

    with pytest.raises(ValueError, match="same number of classes"):
        _validate_classification_labels(y)


def test_tabiclv2_many_classes_rejects_cache() -> None:
    model = TabICLv2(pretrained=False)
    native_x = torch.randn(5, 6)
    native_y = torch.tensor([0, 1, 2, 0, 1])
    model.fit(native_x, native_y)
    assert model.predict(torch.randn(1, 6)).size(-1) == 3

    x = torch.randn(11, 6)
    y = torch.arange(11)

    with pytest.raises(
        NotImplementedError,
        match="caching is not supported with more than 10 classes",
    ):
        model.fit(x, y)

    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(torch.randn(2, 6))


def test_class_permutations() -> None:
    shifts = _class_permutations(
        num_classes=3,
        n_estimators=3,
        method="shift",
        generator=None,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        shifts,
        torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]]),
    )

    first = _class_permutations(
        num_classes=4,
        n_estimators=8,
        method="random",
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
    )
    second = _class_permutations(
        num_classes=4,
        n_estimators=8,
        method="random",
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first, second)
    assert first.unique(dim=0).size(0) == first.size(0)


def test_classification_rejects_complex_targets() -> None:
    model = TabICLv2(pretrained=False)
    x = torch.randn(3, 2)
    y = torch.tensor([0 + 1j, 1 + 0j])

    with pytest.raises(TypeError, match="integer targets"):
        model.predict_proba(x, y)
    with pytest.raises(TypeError, match="real-valued targets"):
        model(x, y)


def test_hierarchical_requires_two_native_classes() -> None:
    with pytest.raises(ValueError, match="at least two native classes"):
        _balanced_grouping(
            num_classes=2,
            max_classes=1,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="at least two native classes"):
        _mixed_radix_bases(num_classes=2, max_classes=1)

    model = _make_small_classifier(
        max_classes=1,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="at least two native classes"):
        model(torch.randn(3, 6), torch.arange(2))


def test_class_shuffled_logit_ensemble() -> None:
    member_probabilities = torch.tensor(
        [
            [[0.6, 0.3, 0.1]],
            [[0.2, 0.5, 0.3]],
            [[0.1, 0.2, 0.7]],
        ]
    )

    class Classifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.targets: list[torch.Tensor] = []

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            member = len(self.targets)
            permutation = y
            self.targets.append(y.clone())
            original_logits = 0.9 * member_probabilities[member].log()
            return original_logits.index_select(-1, permutation.argsort())

    model = TabICLv2(pretrained=False)
    classifier = Classifier()
    model.cls_model = cast(_TabICLv2, classifier)
    probabilities = model.predict_proba(
        torch.randn(4, 2),
        torch.arange(3),
        n_estimators=3,
        class_shuffle_method="shift",
    )

    expected_targets = (
        torch.tensor([0, 1, 2]),
        torch.tensor([2, 0, 1]),
        torch.tensor([1, 2, 0]),
    )
    assert len(classifier.targets) == len(expected_targets)
    for actual, expected in zip(classifier.targets, expected_targets):
        torch.testing.assert_close(actual, expected)

    expected = member_probabilities.log().mean(dim=0).softmax(dim=-1)
    torch.testing.assert_close(probabilities, expected)
    assert not torch.allclose(
        probabilities,
        member_probabilities.mean(dim=0),
    )


@withCUDA
def test_tabiclv2_many_classes_predict_proba(
    device: torch.device,
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    num_classes, test_size = 11, 2
    x = torch.randn(num_classes + test_size, 6, device=device)
    y = torch.arange(num_classes, dtype=torch.int32, device=device)

    probabilities = model.predict_proba(
        x,
        y,
        n_estimators=2,
    )

    assert probabilities.size() == (test_size, num_classes)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(test_size, device=device),
    )
