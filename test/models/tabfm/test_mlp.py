from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from sdm.models.tabfm.mlp import MLP


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("relu", F.relu(torch.tensor([-1.0]))),
        ("gelu", F.gelu(torch.tensor([-1.0]), approximate="tanh")),
        ("silu", F.silu(torch.tensor([-1.0]))),
    ],
)
def test_mlp_applies_configured_hidden_activation(
    activation: Literal["relu", "gelu", "silu"],
    expected: torch.Tensor,
) -> None:
    module = MLP(
        in_channels=1,
        hidden_channels=[1],
        out_channels=1,
        activation=activation,
    )
    with torch.no_grad():
        for layer in module.layers:
            layer.weight.fill_(1)
            layer.bias.zero_()

    output = module(torch.tensor([[-1.0]]))

    torch.testing.assert_close(output.flatten(), expected)


def test_mlp_preserves_leading_dimensions_without_hidden_layers() -> None:
    module = MLP(
        in_channels=2,
        hidden_channels=[],
        out_channels=3,
    )
    input = torch.randn(2, 4, 2)

    output = module(input)

    assert len(module.layers) == 1
    assert output.shape == (2, 4, 3)
