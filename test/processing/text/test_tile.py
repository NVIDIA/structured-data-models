# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import torch

from sdm import TableTensor
from sdm.processing.text.tile import textify
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
