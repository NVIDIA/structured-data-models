# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from textwrap import dedent

import pytest
import torch

import sdm.processing as sp
from sdm import TableTensor
from sdm.processing.text.tile import MODELS, TILE, textify
from sdm.testing import withCUDA


@withCUDA
def test_textify(device: torch.device) -> None:
    table = TableTensor.from_columns(
        data={
            "description": ["gem 💎", None],
            "amount": [1.5, None],
            "color": ["red", None],
            "when": [datetime(1970, 1, 1), None],
            "key": ["a1", "a2"],
        },
        stypes={
            "description": "text",
            "amount": "numerical",
            "color": "categorical",
            "when": "datetime",
            "key": "id",
        },
        device=device,
    )
    expected = [
        [
            "The amount is 1.5.",
            "The color is red.",
            "The when is 1970-01-01 00:00:00.000000.",
            "The description is gem 💎.",
            "The key is a1.",
        ],
        [
            "The amount is missing.",
            "The color is missing.",
            "The when is missing.",
            "The description is missing.",
            "The key is a2.",
        ],
    ]

    array, span = textify(table)

    assert array.to_pylist() == [" ".join(sentences) for sentences in expected]
    assert span.size() == (2, 5, 2)


@withCUDA
def test_tile(device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sentence_transformers")
    monkeypatch.setitem(MODELS, "tiny", ("yujiepan/qwen3-tiny-random", 1))

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
    encoder = TILE(
        model="tiny",
        postprocess=[sp.Standardize(), sp.PCA(2)],
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )
    assert repr(encoder) == dedent("""\
        TILE(
          model='tiny',
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
