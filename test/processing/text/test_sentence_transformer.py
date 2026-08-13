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
        f"{column}__emb{i}"
        for column in ("text_0", "text_1")
        for i in range(128)
    )
    assert output.numerical.shape == (2, 256)
    assert output.numerical.device == device
