# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.timesfm3.configs import ResidualBlockConfig
from sdm.models.timesfm3.dense import ResidualBlock
from sdm.testing import withCUDA


@withCUDA
def test_residual_block_loads_checkpoint_weights(device: torch.device) -> None:
    config = ResidualBlockConfig(
        hidden_dims=2,
        output_dims=2,
        use_bias=False,
        activation="relu",
    )
    block = ResidualBlock(config, input_dims=3, device=device)
    block.load_state_dict(
        {
            "hidden_layer.weight": torch.tensor(
                [[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]],
                device=device,
            ),
            "output_layer.weight": torch.tensor(
                [[1.0, 2.0], [-1.0, 1.0]],
                device=device,
            ),
            "residual_layer.weight": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                device=device,
            ),
        }
    )

    x = torch.tensor([[2.0, 1.0, 3.0]], device=device)

    output = block(x)

    torch.testing.assert_close(output, output.new_tensor([[11.0, 6.0]]))


@withCUDA
def test_residual_block_explicitly_sets_input_dimension(
    device: torch.device,
) -> None:
    config = ResidualBlockConfig(
        hidden_dims=3,
        output_dims=4,
        use_bias=True,
        activation="swish",
    )
    block = ResidualBlock(config, input_dims=5).to(
        device=device, dtype=torch.float64
    )
    parameter_ids = tuple(map(id, block.parameters()))
    x = torch.randn(2, 5, device=device, dtype=torch.float64)

    output = block(x)
    assert tuple(map(id, block.parameters())) == parameter_ids

    assert output.shape == (2, 4)
    assert output.device == device
    assert output.dtype == x.dtype


def test_residual_block_meta_device() -> None:
    config = ResidualBlockConfig(
        hidden_dims=3,
        output_dims=4,
        use_bias=True,
        activation="swish",
        prenorm="rms",
    )
    block = ResidualBlock(config, input_dims=5, device="meta")
    assert all(
        parameter.device.type == "meta" for parameter in block.parameters()
    )


@withCUDA
def test_residual_block_identity_skip(device: torch.device) -> None:
    config = ResidualBlockConfig(
        hidden_dims=3,
        output_dims=2,
        use_bias=False,
        activation="none",
        identity_skip=True,
    )
    block = ResidualBlock(config, input_dims=2, device=device)
    with torch.no_grad():
        block.hidden_layer.weight.zero_()
        block.output_layer.weight.zero_()
    x = torch.tensor([[1.0, -2.0]], device=device)

    output = block(x)

    torch.testing.assert_close(output, x)


def test_residual_block_identity_skip_requires_matching_dimensions() -> None:
    config = ResidualBlockConfig(
        hidden_dims=3,
        output_dims=1,
        use_bias=False,
        activation="none",
        identity_skip=True,
    )

    with pytest.raises(ValueError, match="identity_skip requires"):
        ResidualBlock(config, input_dims=4)


@withCUDA
def test_residual_block_rms_prenorm(device: torch.device) -> None:
    config = ResidualBlockConfig(
        hidden_dims=2,
        output_dims=2,
        use_bias=False,
        activation="none",
        identity_skip=True,
        prenorm="rms",
    )
    block = ResidualBlock(config, input_dims=2, device=device)
    with torch.no_grad():
        block.hidden_layer.weight.copy_(torch.eye(2, device=device))
        block.output_layer.weight.copy_(torch.eye(2, device=device))
    x = torch.tensor([[3.0, 4.0]], device=device)

    output = block(x)

    expected = x + torch.nn.functional.rms_norm(x, (2,))
    torch.testing.assert_close(output, expected)
