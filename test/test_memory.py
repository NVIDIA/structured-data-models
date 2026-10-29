# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm._memory import _split_tensor


def _square_selected(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[:, mask]
    return torch.cat(
        [chunk.square() for chunk in _split_tensor(selected, 8, dim=-1)],
        dim=-1,
    )


def test_bounded_symbolic_split_compiles(compilation: None) -> None:
    values = torch.arange(24.0).reshape(3, 8)
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
        compiled = torch.compile(_square_selected, fullgraph=True)
        for width in (0, 1, 3, 8):
            mask = torch.arange(8) < width
            torch.testing.assert_close(
                compiled(values, mask), values[:, mask].square()
            )


def test_split_preserves_chunk_boundaries() -> None:
    values = torch.arange(8)
    chunks = _split_tensor(values, 3, dim=0)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7]]
    assert [chunk.tolist() for chunk in chunks] == expected
