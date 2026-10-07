# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.models.kumo.relational.graph import _coo_to_csr
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compile_csr_changing_size(
    fullgraph: bool, dtype: torch.dtype, device: torch.device
) -> None:
    def convert(indices: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        return _coo_to_csr(
            indices, nodes.size(0), out_int32=dtype == torch.int32
        )

    torch.compiler.reset()
    compiled = torch.compile(convert, fullgraph=fullgraph, dynamic=True)
    for size in (5, 9, 13, 0, 1, 5):
        indices = torch.arange(
            size, dtype=dtype, device=device
        ).repeat_interleave(2)[::2]
        nodes = torch.empty(size, device=device)
        counts = indices.long().bincount(minlength=size)
        expected = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(dtype)
        assert torch.equal(compiled(indices, nodes), expected)
