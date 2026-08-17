from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import SentenceTransformer
from sdm.processing.text.sentence_transformer import _BPETokenizer
from sdm.testing import onlyCUDA, withCUDA


@onlyCUDA
def test_bpe_pre_tokenize_unicode() -> None:
    cudf = pytest.importorskip("cudf")
    text = cudf.Series(
        [
            "héllo café",
            "naïve façade",
            "中文测试",
            "emoji 😀 rocket 🚀",
        ]
    )

    words = text.str.findall(_BPETokenizer._BYTE_LEVEL_PAT)

    assert words.to_arrow().to_pylist() == [
        ["héllo", " café"],
        ["naïve", " façade"],
        ["中文测试"],
        ["emoji", " 😀", " rocket", " 🚀"],
    ]


@onlyCUDA
def test_bpe_translate_utf8_bytes() -> None:
    cudf = pytest.importorskip("cudf")
    tokenizer = _BPETokenizer(
        encoder=cast(Any, None),
        vocab=cast(Any, None),
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
        max_length=512,
    )
    values = ["ASCII", "héllo", "中文 😀", "", "naïve façade"]
    words = cudf.Series(["ignored", *values])[1:]
    byte_encoder = tokenizer._byte_encoder()
    expected = [
        "".join(byte_encoder[byte] for byte in value.encode())
        for value in values
    ]

    translated = tokenizer._translate_bytes(words)

    assert translated.to_arrow().to_pylist() == expected


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
