# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import pyarrow as pa
import pytest
import torch

from sdm.processing.text.llm import LLM


@pytest.mark.parametrize("deduplicate", [False, True])
@pytest.mark.parametrize("return_type", ["token", "sentence"])
def test_forward(
    return_type: Literal["token", "sentence"],
    deduplicate: bool,
) -> None:
    pytest.importorskip("sentence_transformers")
    encoder = LLM(
        model="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        dtype=torch.float16,
    )
    rows = ["hello world", "hello", "hello"]

    offset = mask = None
    iterator = encoder(
        pa.array(rows),
        return_type,
        deduplicate=deduplicate,
        device="cpu",
    )
    if return_type == "token":
        index, emb, offset, mask = next(iterator)
    else:
        index, emb = next(iterator)

    if deduplicate:
        emb = emb[index[0]]
        offset = offset[index[0]] if offset is not None else None
        mask = mask[index[0]] if mask is not None else None
        index = index[1]

    assert index.equal(torch.arange(3))
    assert emb.is_cpu
    assert emb.dtype == torch.float16
    if return_type == "token":
        assert emb.size() == (3, 4, 128)
        assert offset is not None
        assert offset[0].equal(torch.tensor([[0, 0], [0, 5], [6, 11], [0, 0]]))
        assert offset[1].equal(torch.tensor([[0, 0], [0, 5], [0, 0], [0, 0]]))
        assert offset[2].equal(torch.tensor([[0, 0], [0, 5], [0, 0], [0, 0]]))
        assert mask is not None
        assert mask.equal(
            torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, True, False],
                ],
            )
        )
    else:
        assert emb.size() == (3, 128)
