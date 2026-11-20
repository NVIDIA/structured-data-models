# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from textwrap import dedent

import pytest
import torch

import sdm.processing as sp
from sdm import TableTensor
from sdm.testing import withCUDA


@withCUDA
def test_sentence_transformer(device: torch.device) -> None:
    pytest.importorskip("sentence_transformers")

    table = TableTensor.from_columns(
        data={
            "description": ["gem 💎", "polished stone", "blue", "red"],
            "origin": ["Australia", "Brazil", "India", "Ghana"],
            "amount": [1.5, 2.5, 3.5, 4.5],
        },
        stypes={
            "description": "text",
            "origin": "text",
            "amount": "numerical",
        },
        device=device,
    )
    encoder = sp.SentenceTransformer(
        model="sentence-transformers-testing/stsb-bert-tiny-safetensors",
        postprocess=[sp.Standardize(), sp.PCA(2)],
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )
    assert encoder.requires_fit
    assert repr(encoder) == dedent("""\
        SentenceTransformer(
          model='sentence-transformers-testing/stsb-bert-tiny-safetensors',
          postprocess=Sequential(
            Standardize(),
            PCA(num_components=2, solver='auto'),
          ),
        )""")

    encoder.fit(table[:3])
    out = encoder.transform(table)

    assert out.size() == (4, 5)
    assert out.device == table.device
    assert out.dtype == table.dtype
    assert out.text_columns == ()
    assert out.numerical_columns == (
        "amount",
        "description__emb0",
        "description__emb1",
        "origin__emb0",
        "origin__emb1",
    )
    assert out.numerical.isfinite().all()
    torch.testing.assert_close(out.numerical[:, :1], table.numerical)
    torch.testing.assert_close(
        encoder.fit_transform(table[:3]).numerical,
        out.numerical[:3],
    )
