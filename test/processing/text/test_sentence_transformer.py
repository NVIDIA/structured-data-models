from __future__ import annotations

import importlib.util
from typing import Any

import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import SentenceTransformer
from sdm.processing.text.sentence_transformer import _CuDFTokenizer, _Encoder
from sdm.testing import onlyCUDA, withCUDA

MODEL_NAME = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture(scope="module")
def sentence_transformer_model() -> Any:
    st = pytest.importorskip("sentence_transformers")
    return st.SentenceTransformer(MODEL_NAME)


def test_cudf_tokenizer_supports_canonical_bert(
    sentence_transformer_model: Any,
) -> None:
    assert _CuDFTokenizer._is_supported(sentence_transformer_model)


def test_cudf_tokenizer_supports_configured_word_length_limit(
    sentence_transformer_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wordpiece = sentence_transformer_model.tokenizer.backend_tokenizer.model
    monkeypatch.setattr(wordpiece, "max_input_chars_per_word", 64)

    assert _CuDFTokenizer._is_supported(sentence_transformer_model)


@pytest.mark.parametrize("component", ["normalizer", "pre_tokenizer"])
def test_cudf_tokenizer_rejects_different_pipeline(
    component: str,
    sentence_transformer_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    replacements = {
        "normalizer": tokenizers.normalizers.NFKC(),
        "pre_tokenizer": tokenizers.pre_tokenizers.WhitespaceSplit(),
    }
    backend = sentence_transformer_model.tokenizer.backend_tokenizer
    monkeypatch.setattr(backend, component, replacements[component])

    assert not _CuDFTokenizer._is_supported(sentence_transformer_model)


@pytest.mark.parametrize(
    ("target", "attribute", "value"),
    [
        ("model", "default_prompt_name", "query"),
        ("model", "truncate_dim", 64),
        ("module", "processing_kwargs", {"text": {"max_length": 32}}),
        ("tokenizer", "truncation_side", "left"),
    ],
)
def test_cudf_tokenizer_rejects_bypassed_encode_options(
    target: str,
    attribute: str,
    value: object,
    sentence_transformer_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owners = {
        "model": sentence_transformer_model,
        "module": sentence_transformer_model[0],
        "tokenizer": sentence_transformer_model.tokenizer,
    }
    monkeypatch.setattr(owners[target], attribute, value)

    assert not _CuDFTokenizer._is_supported(sentence_transformer_model)


def test_unsupported_tokenizer_does_not_probe_cudf(
    sentence_transformer_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    backend = sentence_transformer_model.tokenizer.backend_tokenizer
    monkeypatch.setattr(backend, "normalizer", tokenizers.normalizers.NFKC())

    def fail_if_called(name: str) -> None:
        raise AssertionError(f"Unexpected dependency probe for {name}")

    monkeypatch.setattr(importlib.util, "find_spec", fail_if_called)

    assert _CuDFTokenizer.build(sentence_transformer_model) is None


@onlyCUDA
def test_cudf_wordpiece_byte_limit() -> None:
    cudf = pytest.importorskip("cudf")
    wordpiece = pytest.importorskip("cudf.core.wordpiece_tokenize")

    vocab = [
        *(f"[unused{i}]" for i in range(100)),
        "[UNK]",
        "a",
        "##a",
    ]
    tokenizer = wordpiece.WordPieceVocabulary(cudf.Series(vocab))
    below_limit, at_limit = (
        tokenizer.tokenize(cudf.Series(["a" * 199, "a" * 200]))
        .to_arrow()
        .to_pylist()
    )

    assert below_limit == [101] + [102] * 198
    assert at_limit == [100]


@onlyCUDA
def test_gpu_routes_word_length_mismatches(
    sentence_transformer_model: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cudf")
    device = torch.device("cuda:0")
    wordpiece = sentence_transformer_model.tokenizer.backend_tokenizer.model
    monkeypatch.setattr(wordpiece, "max_input_chars_per_word", 128)

    encoder = _Encoder(
        sentence_transformer_model,
        batch_size=32,
        embedding_dim=1,
    )
    tokenizer = encoder._cudf_tokenizer
    assert tokenizer is not None
    assert tokenizer._max_input_chars_per_word == 128

    def forward_cpu(text: StringTensor) -> torch.Tensor:
        return torch.full(
            (text.numel(), 1),
            -1,
            dtype=torch.float32,
            device=text.device,
        )

    def forward_cudf(text: StringTensor) -> torch.Tensor:
        return torch.full(
            (text.numel(), 1),
            1,
            dtype=torch.float32,
            device=text.device,
        )

    monkeypatch.setattr(encoder, "_forward_cpu", forward_cpu)
    monkeypatch.setattr(encoder, "_forward_cudf", forward_cudf)

    cpu_rejects = "a" * 129
    cudf_rejects = "\N{CYRILLIC SMALL LETTER BE}" * 100
    both_reject = "\N{CYRILLIC SMALL LETTER BE}" * 129
    cases = [
        (["short", both_reject], [1, 1]),
        ([cpu_rejects, cudf_rejects], [-1, -1]),
        (
            ["short", cpu_rejects, cudf_rejects, both_reject],
            [1, -1, -1, 1],
        ),
    ]

    for texts, expected in cases:
        output = encoder(StringTensor.from_list(texts, device=device))
        torch.testing.assert_close(
            output.flatten(),
            torch.tensor(expected, dtype=torch.float32, device=device),
        )


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
    checksum = "0123456789abcdef" * 8
    cyrillic = "\N{CYRILLIC SMALL LETTER BE}" * 100
    text_batches = [
        [
            ["hello world", "short"],
            ["", "a"],
            [None, "café naïve üñîçødé"],
            [long_text, "x y z"],
            [checksum, cyrillic],
        ],
        [
            [checksum, cyrillic],
            [cyrillic, checksum],
        ],
    ]

    model.eval()
    for texts in text_batches:
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
