# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.norm import ColumnRMSNorm, RowRMSNorm
from sdm.nn import LogScale
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


@withCUDA
def test_transformer_block(device: torch.device) -> None:
    block = KumoTabularTransformerBlock(
        channels=32,
        num_heads=4,
        query_scaling=LogScale(num_heads=4, device=device),
        device=device,
    )
    query = torch.randn(2, 5, 32, device=device)
    key_value = torch.randn(2, 3, 32, device=device)

    output = block(query=query, key_value=key_value)

    torch.testing.assert_close(output, query, rtol=0, atol=0)


@onlyCUDA
@onlyFullTest
def test_column_rms_norm_compile_dynamic_shapes() -> None:
    torch._dynamo.reset()
    norm = ColumnRMSNorm(128, device="cuda")

    def make_input(columns: int, rows: int) -> torch.Tensor:
        return torch.randn(
            rows,
            columns,
            128,
            dtype=torch.float16,
            device="cuda",
        ).transpose(0, 1)

    try:
        with torch.inference_mode(), torch.autocast("cuda", torch.float16):
            x = make_input(columns=10, rows=4_000)
            expected = torch.nn.functional.rms_norm(
                x,
                norm.normalized_shape,
                weight=norm.weight,
                eps=norm.eps,
            )
            torch.testing.assert_close(norm(x), expected)

            with torch._dynamo.config.patch(error_on_recompile=True):
                for columns, rows in [(100, 10_000), (10, 100_000)]:
                    norm(make_input(columns, rows))
    finally:
        torch._dynamo.reset()


@onlyCUDA
@onlyFullTest
def test_row_rms_norm_compile_dynamic_shapes() -> None:
    torch._dynamo.reset()
    norm = RowRMSNorm(128, device="cuda")

    def make_input(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(
            rows,
            columns + 4,
            128,
            dtype=torch.float16,
            device="cuda",
        )

    try:
        with torch.inference_mode(), torch.autocast("cuda", torch.float16):
            x = make_input(rows=4_000, columns=10)
            expected = torch.nn.functional.rms_norm(
                x,
                norm.normalized_shape,
                weight=norm.weight,
                eps=norm.eps,
            )
            torch.testing.assert_close(norm(x), expected)

            with torch._dynamo.config.patch(error_on_recompile=True):
                for rows, columns in [(10_000, 100), (2_000, 500)]:
                    norm(make_input(rows, columns))
    finally:
        torch._dynamo.reset()
