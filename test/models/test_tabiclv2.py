from typing import cast

import pytest
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sdm.models.tabiclv2.model import _TabICLv2
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import Attention
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


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
# @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])  # TODO Reenable
@pytest.mark.parametrize("batch_shape", [()])
def test_forward(
    device: torch.device,
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "TabICLv2()"
    else:
        assert repr(model) == "TabICLv2(device=cuda:0)"

    R_context, R_query, C = 5, 3, 6

    x_context = torch.randn(*batch_shape, R_context, C, device=device)
    x_query = torch.randn(*batch_shape, R_query, C, device=device)
    if dtype.is_floating_point:
        y_context = torch.randn((*batch_shape, R_context, 1), device=device)
        out = model(x_context, y_context, x_query)
        assert out.size() == (1, *batch_shape, R_query, 999)
    else:
        y_context = torch.randint(
            low=0,
            high=10,
            size=(*batch_shape, R_context, 1),
            device=device,
        )
        num_classes = len(y_context.unique())
        out = model(x_context, y_context, x_query)
        assert out.size() == (1, *batch_shape, R_query, num_classes)

    assert out.dtype == x_query.dtype
    assert out.device == x_query.device
    assert torch.is_inference(out)

    if len(batch_shape) > 0:
        looped = torch.stack(
            [
                model(x_context[i], y_context[i], x_query[i])
                for i in range(batch_shape[0])
            ],
            dim=0,
        )
        torch.testing.assert_close(
            out.numerical, cast(TableTensor, looped).numerical
        )

    model.fit(x_context, y_context)
    torch.testing.assert_close(model.predict(x_query).numerical, out.numerical)
    model.clear()


# @pytest.mark.parametrize("batch_shape", [(), (2,)])  # TODO Reenable
@pytest.mark.parametrize("batch_shape", [()])
def test_num_estimators(batch_shape: tuple[int, ...]) -> None:
    model = TabICLv2(pretrained=False)

    R_context, R_query, C = 5, 3, 6

    x_context = torch.randn(*batch_shape, R_context, C)
    x_query = torch.randn(*batch_shape, R_query, C)
    y_context = torch.randn(*batch_shape, R_context, 1)

    out = model(x_context, y_context, x_query, num_estimators=2)
    assert out.size() == (2, *batch_shape, R_query, 999)

    model.fit(x_context, y_context, num_estimators=3)
    out = model.predict(x_query)
    assert out.size() == (3, *batch_shape, R_query, 999)
    model.clear()


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
    native_x_context = torch.randn(5, 6)
    native_y_context = torch.tensor([[0], [1], [2], [0], [1]])
    model.fit(native_x_context, native_y_context)
    assert model.predict(torch.randn(1, 6)).size(-1) == 3

    x_context = torch.randn(11, 6)
    y_context = torch.arange(11).unsqueeze(-1)

    with pytest.raises(
        NotImplementedError,
        match="caching is not supported with more than 10 classes",
    ):
        model.fit(x_context, y_context)

    with pytest.raises(RuntimeError, match="not yet fitted"):
        model.predict(torch.randn(2, 6))


@withCUDA
def test_tabiclv2_many_classes_forward(device: torch.device) -> None:
    model = TabICLv2(pretrained=False, device=device)
    num_classes, test_size = 11, 2
    x_context = torch.randn(num_classes, 6, device=device)
    x_query = torch.randn(test_size, 6, device=device)
    y_context = torch.arange(
        num_classes,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(-1)

    out = model(x_context, y_context, x_query)

    assert out.size() == (1, test_size, num_classes)
    probabilities = out.numerical
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(1, test_size, device=device),
    )


@onlyCUDA
@onlyFullTest
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_compile(dtype: torch.dtype) -> None:
    torch._dynamo.reset()
    model = TabICLv2(pretrained=False, device="cuda")

    R_context, R_query, C = 5, 3, 6
    x_context = torch.randn(R_context, C, device="cuda")
    x_query = torch.randn(R_query, C, device="cuda")

    if dtype.is_floating_point:
        y_context = torch.randn(R_context, 1, device="cuda")
    else:
        y_context = torch.randint(0, 10, size=(R_context, 1), device="cuda")

    expected = model(x_context, y_context, x_query)
    submodel = model.reg_model if dtype.is_floating_point else model.cls_model
    submodel.compile(fullgraph=True)

    actual = model(x_context, y_context, x_query)
    torch.testing.assert_close(actual, expected)
    assert torch.is_inference(actual)

    model.fit(x_context, y_context)
    predicted = model.predict(x_query)
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
