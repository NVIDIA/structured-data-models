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
        in_channels=3,
        out_channels=2,
        bias=False,
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
        in_channels=1,
        out_channels=1,
        bias=False,
        identity_residual=True,
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
        in_channels=5,
        out_channels=4,
        bias=True,
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
        in_channels=5,
        out_channels=4,
        bias=True,
        prenorm=True,
        device="meta",
    )
    assert all(
        parameter.device.type == "meta" for parameter in block.parameters()
    )


@withCUDA
def test_residual_block_identity_residual(device: torch.device) -> None:
    block = ResidualBlock(
        in_channels=2,
        out_channels=2,
        bias=False,
        identity_residual=True,
        device=device,
    )
    with torch.no_grad():
        block.hidden_layer.weight.zero_()
        block.output_layer.weight.zero_()
    x = torch.tensor([[1.0, -2.0]], device=device)

    output = block(x)

    torch.testing.assert_close(output, x)


def test_residual_block_identity_residual_requires_equal_channels() -> None:
    with pytest.raises(ValueError, match="identity_residual requires"):
        ResidualBlock(
            in_channels=4,
            out_channels=1,
            bias=False,
            identity_residual=True,
        )


@withCUDA
def test_residual_block_rms_prenorm(device: torch.device) -> None:
    block = ResidualBlock(
        in_channels=2,
        out_channels=2,
        bias=False,
        identity_residual=True,
        prenorm=True,
        device=device,
    )
    with torch.no_grad():
        block.hidden_layer.weight.copy_(torch.eye(2, device=device))
        block.output_layer.weight.copy_(torch.eye(2, device=device))
    x = torch.tensor([[3.0, 4.0]], device=device)

    output = block(x)

    expected = x + F.rms_norm(x, (2,))
    torch.testing.assert_close(output, expected)
