import inspect

import pytest
import torch
from sdm.models.kumorfm_v2_1 import ICLPredictor, RowEmbedding
from sdm.nn import TransformerBlock
from sdm.testing import withCUDA
from torch import Tensor


def _make_predictor(
    *,
    num_layers: int = 2,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ICLPredictor:
    return ICLPredictor(
        channels=8,
        num_layers=num_layers,
        num_heads=2,
        num_quantiles=11,
        max_classes=4,
        device=device,
        dtype=dtype,
    )


def _enable_context_path(predictor: ICLPredictor) -> None:
    final_layer = predictor.layers[-1]
    assert isinstance(final_layer, TransformerBlock)
    with torch.no_grad():
        torch.nn.init.eye_(final_layer.attn.out_lin.weight)


@pytest.mark.parametrize(
    ("y_train", "output_size"),
    [
        (torch.tensor([0, 3, 1]), 4),
        (torch.tensor([False, True, False]), 4),
        (torch.tensor([0.1, -0.2, 0.3]), 11),
    ],
)
def test_icl_predictor_task_output_shapes(
    y_train: Tensor,
    output_size: int,
) -> None:
    predictor = _make_predictor()
    x = torch.randn(7, 8)

    out = predictor(x, y_train)

    assert out.size() == (4, output_size)
    assert out.dtype == x.dtype
    assert out.device == x.device
    assert torch.isfinite(out).all()


def test_icl_predictor_architecture_defaults() -> None:
    parameters = inspect.signature(ICLPredictor).parameters

    assert parameters["channels"].default == 512
    assert parameters["num_layers"].default == 12
    assert parameters["num_heads"].default == 8
    assert parameters["num_quantiles"].default == 999
    assert parameters["max_classes"].default == 10

    predictor = _make_predictor()
    assert all(
        isinstance(layer, TransformerBlock) for layer in predictor.layers
    )


class _RecordingLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Tensor, Tensor]] = []

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        self.calls.append((query.detach().clone(), key_value.detach().clone()))
        return query + key_value.mean(dim=0, keepdim=True)


def test_every_layer_uses_context_and_final_layer_queries_tests() -> None:
    predictor = _make_predictor(num_layers=3)
    layers = [_RecordingLayer() for _ in range(3)]
    predictor.layers = torch.nn.ModuleList(layers)
    x = torch.randn(6, 8)
    y_train = torch.tensor([0, 2])

    out = predictor(x, y_train)

    assert out.size() == (4, 4)
    assert [layer.calls[0][0].size(0) for layer in layers] == [6, 6, 4]
    assert [layer.calls[0][1].size(0) for layer in layers] == [2, 2, 2]
    torch.testing.assert_close(
        layers[1].calls[0][1],
        layers[0].calls[0][0][:2]
        + layers[0].calls[0][1].mean(dim=0, keepdim=True),
    )


def test_test_rows_do_not_leak_into_other_predictions() -> None:
    torch.manual_seed(0)
    predictor = _make_predictor(num_layers=2)
    predictor.layers = torch.nn.ModuleList(
        [_RecordingLayer(), _RecordingLayer()]
    )
    x = torch.randn(6, 8)
    y_train = torch.tensor([0, 1])

    expected = predictor(x, y_train)
    changed = x.clone()
    changed[-1] = torch.randn_like(changed[-1]) * 1_000
    actual = predictor(changed, y_train)

    torch.testing.assert_close(actual[:-1], expected[:-1])


def test_non_prefix_context_mask_preserves_test_order() -> None:
    predictor = _make_predictor()
    x = torch.randn(6, 8)
    y_train = torch.tensor([0, 2, 1])
    context_mask = torch.tensor([False, True, False, True, True, False])
    permutation = torch.cat(
        [
            context_mask.nonzero(as_tuple=True)[0],
            (~context_mask).nonzero(as_tuple=True)[0],
        ]
    )

    actual = predictor(x, y_train, context_mask=context_mask)
    expected = predictor(x.index_select(0, permutation), y_train)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("regression", [False, True])
def test_test_chunks_match_full_prediction(regression: bool) -> None:
    torch.manual_seed(5)
    predictor = _make_predictor(num_layers=3)
    context = torch.randn(3, 8)
    test = torch.randn(5, 8)
    y_train = torch.randn(3) if regression else torch.tensor([0, 2, 1])

    expected = predictor(torch.cat([context, test]), y_train)
    chunks = [
        predictor(torch.cat([context, chunk]), y_train)
        for chunk in test.split([2, 3])
    ]

    torch.testing.assert_close(torch.cat(chunks), expected)


@withCUDA
@pytest.mark.parametrize("regression", [False, True])
def test_icl_predictor_dtype_device_gradients_and_input_immutability(
    device: torch.device,
    regression: bool,
) -> None:
    torch.manual_seed(1)
    predictor = _make_predictor(device=device, dtype=torch.float64)
    _enable_context_path(predictor)
    x = torch.randn(
        6,
        8,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    x_before = x.detach().clone()
    if regression:
        y_train = torch.randn(
            3,
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
    else:
        y_train = torch.tensor([0, 2, 1], device=device)
    y_before = y_train.detach().clone()

    out = predictor(x, y_train)
    (out * torch.randn_like(out)).sum().backward()

    assert out.dtype == x.dtype
    assert out.device == x.device
    torch.testing.assert_close(x.detach(), x_before, rtol=0, atol=0)
    torch.testing.assert_close(y_train.detach(), y_before, rtol=0, atol=0)
    assert x.grad is not None
    assert x.grad[:3].abs().sum().item() > 0
    assert x.grad[3:].abs().sum().item() > 0
    if regression:
        assert y_train.grad is not None
        assert y_train.grad.abs().sum().item() > 0
        assert predictor.y_reg_lin.weight.grad is not None
        assert predictor.reg_head.weight.grad is not None
    else:
        assert predictor.y_cls_emb.weight.grad is not None
        assert predictor.cls_head.weight.grad is not None


def test_icl_predictor_allows_no_test_rows() -> None:
    predictor = _make_predictor()
    x = torch.randn(3, 8)

    classification = predictor(x, torch.tensor([0, 1, 2]))
    regression = predictor(x, torch.randn(3))

    assert classification.size() == (0, 4)
    assert regression.size() == (0, 11)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"channels": 0}, ValueError, "`channels` must be at least 1"),
        ({"num_layers": 0}, ValueError, "`num_layers` must be at least 1"),
        ({"num_heads": 0}, ValueError, "`num_heads` must be at least 1"),
        (
            {"num_quantiles": 0},
            ValueError,
            "`num_quantiles` must be at least 1",
        ),
        (
            {"max_classes": 0},
            ValueError,
            "`max_classes` must be at least 1",
        ),
        (
            {"channels": 7, "num_heads": 2},
            ValueError,
            "must be divisible",
        ),
        ({"num_layers": True}, TypeError, "must be an integer"),
    ],
)
def test_icl_predictor_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        ICLPredictor(**kwargs)  # ty: ignore[invalid-argument-type]


def test_icl_predictor_rejects_malformed_inputs() -> None:
    predictor = _make_predictor()
    x = torch.randn(4, 8)
    y_train = torch.tensor([0, 1])

    with pytest.raises(ValueError, match=r"`x` must have shape"):
        predictor(x.unsqueeze(0), y_train)
    with pytest.raises(ValueError, match=r"`x` must have 8 channels"):
        predictor(x[:, :7], y_train)
    with pytest.raises(TypeError, match=r"`x` must be a floating-point"):
        predictor(x.long(), y_train)
    with pytest.raises(ValueError, match=r"`y_train` must have shape"):
        predictor(x, y_train.unsqueeze(-1))
    with pytest.raises(TypeError, match=r"`y_train` must contain"):
        predictor(x, torch.tensor([1 + 2j]))
    with pytest.raises(ValueError, match="at least one context row"):
        predictor(x, torch.empty(0))
    with pytest.raises(ValueError, match="more targets than rows"):
        predictor(x, torch.zeros(5))
    with pytest.raises(ValueError, match="Classification labels"):
        predictor(x, torch.tensor([-1]))
    with pytest.raises(ValueError, match="Classification labels"):
        predictor(x, torch.tensor([4]))
    with pytest.raises(ValueError, match=r"`context_mask` must have shape"):
        predictor(
            x,
            y_train,
            context_mask=torch.ones(3, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match=r"`context_mask` must have boolean"):
        predictor(x, y_train, context_mask=torch.ones(4, dtype=torch.long))
    with pytest.raises(ValueError, match="target count must match"):
        predictor(x, y_train, context_mask=torch.ones(4, dtype=torch.bool))


def test_icl_predictor_supports_autocast_composition() -> None:
    row_embedding = RowEmbedding(
        channels=8,
        num_layers=1,
        num_heads=2,
        num_inducing_points=3,
        num_readout_tokens=1,
    )
    predictor = _make_predictor()
    x = torch.randn(4, 8)
    y_train = torch.tensor([0, 1])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        row_embeddings = row_embedding(x, y_train)
        out = predictor(row_embeddings, y_train)

    assert out.dtype == torch.bfloat16


def test_icl_predictor_rejects_device_mismatches() -> None:
    x = torch.randn(4, 8)
    y_train = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="same device as the module"):
        _make_predictor(device="meta")(x, y_train)
    with pytest.raises(ValueError, match="must be on the same device"):
        _make_predictor()(x, y_train.to(device="meta"))
