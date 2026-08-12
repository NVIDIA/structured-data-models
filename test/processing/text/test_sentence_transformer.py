from __future__ import annotations

import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import SentenceTransformer
from sdm.testing import onlyCUDA, withCUDA

MODEL_NAME = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


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

    output = SentenceTransformer(MODEL_NAME).to(device)(table)

    assert output.columns[Stype.numerical] == tuple(
        f"{column}__emb{i}"
        for column in ("text_0", "text_1")
        for i in range(128)
    )
    assert output.numerical.shape == (2, 256)
    assert output.numerical.device == device


@onlyCUDA
def test_gpu_matches_encode() -> None:
    st = pytest.importorskip("sentence_transformers")
    pytest.importorskip("cudf")

    device = torch.device("cuda:0")
    model = st.SentenceTransformer(MODEL_NAME)
    processor = SentenceTransformer(MODEL_NAME).to(device)

    long_text = "word " * model.max_seq_length

    texts = [
        ["hello world", "short"],
        ["", "a"],
        [None, "café naïve üñîçødé"],
        [long_text, "x y z"],
    ]

    table = TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(texts, device=device),
    )
    gpu_output = processor(table)
    gpu_emb = gpu_output.numerical

    flat_texts = [
        t if t is not None else ""
        for col_idx in range(2)
        for row in texts
        for t in [row[col_idx]]
    ]
    model.eval()
    with torch.inference_mode():
        ref_emb = model.encode(
            flat_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=str(device),
        )
    ref_emb = (
        ref_emb.reshape(2, len(texts), -1)
        .movedim(0, -2)
        .reshape(len(texts), -1)
    )

    assert torch.allclose(gpu_emb, ref_emb, atol=1e-5)


@onlyCUDA
def test_gpu_truncation_boundary() -> None:
    st = pytest.importorskip("sentence_transformers")
    pytest.importorskip("cudf")

    device = torch.device("cuda:0")
    model = st.SentenceTransformer(MODEL_NAME)
    processor = SentenceTransformer(MODEL_NAME).to(device)
    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length - 2  # room for [CLS] and [SEP]

    token_ids = list(range(1000, 1000 + max_tokens + 1))
    words = tokenizer.convert_ids_to_tokens(token_ids)

    at_limit = " ".join(words[:max_tokens])
    over_limit = " ".join(words[: max_tokens + 1])

    for text in [at_limit, over_limit]:
        table = TableTensor(
            columns={"text": ("col",)},
            text=StringTensor.from_list([[text]], device=device),
        )
        gpu_emb = processor(table).numerical

        model.eval()
        with torch.inference_mode():
            ref_emb = model.encode(
                [text],
                convert_to_tensor=True,
                show_progress_bar=False,
                device=str(device),
            )

        assert torch.allclose(gpu_emb, ref_emb, atol=1e-5)
