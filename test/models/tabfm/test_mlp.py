from typing import cast

import pytest
import torch
from torch.nn import Linear

from sdm.models.tabfm.mlp import _MLP
from sdm.testing import withCUDA


def _set_golden_state(module: _MLP) -> None:
    with torch.no_grad():
        for index, child in enumerate(module.layers):
            layer = cast(Linear, child)
            values = torch.arange(
                layer.weight.numel(),
                device=layer.weight.device,
                dtype=torch.float32,
            ).view_as(layer.weight)
            layer.weight.copy_(
                (values - (layer.weight.numel() - 1) / 2)
                * (0.17 + index * 0.04)
            )
            layer.bias.copy_(
                torch.linspace(
                    -0.13,
                    0.19,
                    layer.out_features,
                    device=layer.bias.device,
                    dtype=torch.float32,
                )
                * (index + 1)
            )


def _golden_input(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            [[-1.7, 0.2, 1.1], [0.3, -0.8, 2.4]],
            [[0.7, 1.3, -0.4], [-2.1, 0.05, 0.9]],
        ],
        device=device,
        dtype=dtype,
    )


@withCUDA
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (
            torch.float32,
            [
                -0.4189112782,
                0.8197572827,
                -0.5978949070,
                1.3558471203,
                -0.4848102927,
                0.9627038240,
                -0.4206038415,
                0.7673491240,
            ],
        ),
        (
            torch.bfloat16,
            [
                -0.419921875,
                0.8203125,
                -0.59765625,
                1.359375,
                -0.486328125,
                0.96484375,
                -0.419921875,
                0.765625,
            ],
        ),
    ],
)
def test_mlp_matches_google_golden(
    device: torch.device,
    dtype: torch.dtype,
    expected: list[float],
) -> None:
    module = _MLP(
        in_channels=3,
        hidden_channels=(4, 2),
        out_channels=2,
        device=device,
        dtype=dtype,
    )
    _set_golden_state(module)

    output = module(_golden_input(device=device, dtype=dtype))

    # Frozen from google-research/tabfm@b8a8b090's PyTorch MLP.
    torch.testing.assert_close(
        output.flatten(),
        torch.tensor(expected, device=device, dtype=dtype),
    )
    assert output.shape == (2, 2, 2)
    assert output.device == device
    assert output.dtype == dtype


def test_mlp_matches_google_golden_without_hidden_layers() -> None:
    module = _MLP(
        in_channels=3,
        hidden_channels=(),
        out_channels=2,
    )
    _set_golden_state(module)

    output = module(_golden_input())

    expected = torch.tensor(
        [
            [
                [0.4480000138, 0.5640000105],
                [-0.2575000226, 1.0315001011],
            ],
            [
                [-0.7250000238, 0.4109999537],
                [0.6732499600, 0.4067500234],
            ],
        ]
    )
    torch.testing.assert_close(output, expected)
    assert set(module.state_dict()) == {"layers.0.weight", "layers.0.bias"}


def test_mlp_uses_checkpoint_state_names() -> None:
    module = _MLP(
        in_channels=3,
        hidden_channels=(4, 2),
        out_channels=2,
    )

    assert set(module.state_dict()) == {
        "layers.0.weight",
        "layers.0.bias",
        "layers.1.weight",
        "layers.1.bias",
        "layers.2.weight",
        "layers.2.bias",
    }


def test_mlp_compiles_and_backpropagates() -> None:
    module = _MLP(
        in_channels=3,
        hidden_channels=(4, 2),
        out_channels=2,
    )
    with torch.no_grad():
        for child in module.layers:
            layer = cast(Linear, child)
            layer.weight.fill_(0.2)
            layer.bias.fill_(0.1)
    x = torch.tensor([[[-1.0, 0.5, 2.0]]], requires_grad=True)
    with torch.no_grad():
        expected = module(x)
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    output = compiled(x)
    torch.testing.assert_close(output, expected)
    output.sum().backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad) == x.numel()
    for parameter in module.parameters():
        assert parameter.grad is not None
