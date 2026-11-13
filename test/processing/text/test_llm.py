# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
import torch

from sdm.processing.text.llm import LLM


def test_forward() -> None:
    pytest.importorskip("sentence_transformers")
    encoder = LLM(
        model="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        dtype=torch.float16,
    )
    rows = ["hello world", "hello", "world"]

    perm, emb, offset, mask = next(iter(encoder(pa.array(rows), device="cpu")))

    assert perm.equal(torch.arange(3))
    assert emb.size() == (3, 4, 128)
    assert emb.is_cpu
    assert emb.dtype == torch.float16
    assert offset[0].equal(torch.tensor([[0, 0], [0, 5], [6, 11], [0, 0]]))
    assert offset[1].equal(torch.tensor([[0, 0], [0, 5], [0, 0], [0, 0]]))
    assert offset[2].equal(torch.tensor([[0, 0], [0, 5], [0, 0], [0, 0]]))
    assert mask.equal(
        torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, False],
                [True, True, True, False],
            ],
        )
    )
