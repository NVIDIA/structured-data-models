# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from sdm.models.timesfm3.dense import ResidualBlock
from sdm.testing import withCUDA


@withCUDA
def test_residual_block_loads_checkpoint_weights(device: torch.device) -> None:
    block = ResidualBlock(
        input_dims=3,
        hidden_dims=2,
        output_dims=2,
        use_bias=False,
        device=device,
    )
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
def test_residual_block_uses_relu(device: torch.device) -> None:
    block = ResidualBlock(
        input_dims=1,
        hidden_dims=1,
        output_dims=1,
        use_bias=False,
        identity_skip=True,
        device=device,
    )
    with torch.no_grad():
        block.hidden_layer.weight.fill_(1)
        block.output_layer.weight.fill_(1)
    x = torch.tensor([[-2.0], [3.0]], device=device)

    torch.testing.assert_close(
        block(x), torch.tensor([[-2.0], [6.0]], device=device)
    )


@withCUDA
def test_residual_block_explicitly_sets_input_dimension(
    device: torch.device,
) -> None:
    block = ResidualBlock(
        input_dims=5,
        hidden_dims=3,
        output_dims=4,
        use_bias=True,
    ).to(
        device=device,
        dtype=torch.float64,
    )
    parameter_ids = tuple(map(id, block.parameters()))
    x = torch.randn(2, 5, device=device, dtype=torch.float64)

    output = block(x)
    assert tuple(map(id, block.parameters())) == parameter_ids

    assert output.shape == (2, 4)
    assert output.device == device
    assert output.dtype == x.dtype


def test_residual_block_meta_device() -> None:
    block = ResidualBlock(
        input_dims=5,
        hidden_dims=3,
        output_dims=4,
        use_bias=True,
        prenorm="rms",
        device="meta",
    )
    assert all(
        parameter.device.type == "meta" for parameter in block.parameters()
    )


@withCUDA
def test_residual_block_identity_skip(device: torch.device) -> None:
    block = ResidualBlock(
        input_dims=2,
        hidden_dims=3,
        output_dims=2,
        use_bias=False,
        identity_skip=True,
        device=device,
    )
    with torch.no_grad():
        block.hidden_layer.weight.zero_()
        block.output_layer.weight.zero_()
    x = torch.tensor([[1.0, -2.0]], device=device)

    output = block(x)

    torch.testing.assert_close(output, x)


def test_residual_block_identity_skip_requires_matching_dimensions() -> None:
    with pytest.raises(ValueError, match="identity_skip requires"):
        ResidualBlock(
            input_dims=4,
            hidden_dims=3,
            output_dims=1,
            use_bias=False,
            identity_skip=True,
        )


@withCUDA
def test_residual_block_rms_prenorm(device: torch.device) -> None:
    block = ResidualBlock(
        input_dims=2,
        hidden_dims=2,
        output_dims=2,
        use_bias=False,
        identity_skip=True,
        prenorm="rms",
        device=device,
    )
    with torch.no_grad():
        block.hidden_layer.weight.copy_(torch.eye(2, device=device))
        block.output_layer.weight.copy_(torch.eye(2, device=device))
    x = torch.tensor([[3.0, 4.0]], device=device)

    output = block(x)

    expected = x + F.rms_norm(x, (2,))
    torch.testing.assert_close(output, expected)
