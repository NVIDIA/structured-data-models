from __future__ import annotations

import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import SentenceTransformer
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    pytest.importorskip("sentence_transformers")

    table = TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["a", "b"],
                ["c", None],
            ],
            device=device,
        ),
    )

    output = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    ).to(device)(table)

    assert output.columns[Stype.numerical] == tuple(
        f"{column}_{index}"
        for column in ("title", "body")
        for index in range(128)
    )
    assert output.numerical.shape == (2, 256)
    assert output.numerical.device == device
