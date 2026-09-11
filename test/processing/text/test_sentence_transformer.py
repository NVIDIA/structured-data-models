from __future__ import annotations

import importlib.util
from typing import Any, cast

import pytest
import torch

from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
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

    def forward_cpu(
        text: StringTensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = torch.full(
            (text.numel(), 1),
            -1,
            dtype=torch.float32,
            device=text.device,
        )
        return (
            embeddings
            if out is None
            else out.copy_(embeddings.reshape_as(out))
        )

    def forward_cudf(
        text: StringTensor,
        out: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = torch.full(
            (text.numel(), 1),
            1,
            dtype=torch.float32,
            device=text.device,
        )
        if out is None:
            return embeddings
        assert indices is not None
        out[indices] = embeddings
        return out

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


@withCUDA
@pytest.mark.parametrize("num_numerical", [0, 1])
def test_transform_does_not_track_model_gradients(
    device: torch.device,
    num_numerical: int,
) -> None:
    pytest.importorskip("sentence_transformers")
    processor = SentenceTransformer(MODEL_NAME).to(device)
    assert any(parameter.requires_grad for parameter in processor.parameters())
    table = TableTensor(
        numerical=torch.ones((2, num_numerical), device=device),
        text=StringTensor.from_list([["first"], ["second"]], device=device),
    )

    with torch.enable_grad():
        output = processor(table)

    assert not output.numerical.requires_grad
    assert output.numerical.grad_fn is None


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_batched_mixed_table_preserves_order_and_input(
    device: torch.device,
    dtype: torch.dtype,
    sentence_transformer_model: Any,
) -> None:
    text = StringTensor.from_list(
        [
            [
                ["first", "a longer second text"],
                ["", None],
                ["third", "fourth"],
            ],
            [["another row", "short"], ["last", "last body"], ["x", "y"]],
        ],
        device=device,
    )
    numerical = torch.arange(12, device=device, dtype=dtype).reshape(2, 3, 2)
    categorical = CategoricalTensor(
        code=torch.zeros((2, 3, 1), device=device, dtype=torch.int32),
        categories=(StringTensor.from_list(["category"], device=device),),
    )
    table = cast(
        TableTensor,
        TableTensor(
            columns={"text": ("title", "body")},
            numerical=numerical,
            categorical=categorical,
            datetime=torch.arange(6, device=device).reshape(2, 3, 1),
            text=text,
        ).transpose(0, 1),
    )
    original_numerical = table.numerical.clone()
    original_text = table.text.to_arrow().to_pylist()
    processor = SentenceTransformer(MODEL_NAME, batch_size=2).to(device)
    expected_embeddings = sentence_transformer_model.encode(
        [value or "" for value in original_text],
        batch_size=2,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=str(device),
    ).to(dtype=dtype)
    expected = torch.cat(
        [original_numerical, expected_embeddings.reshape(3, 2, -1)],
        dim=-1,
    )

    output = processor(table)
    assert output.numerical.dtype == dtype
    assert output.columns[Stype.numerical] == (
        *table.columns[Stype.numerical],
        *(
            f"{column}__emb{i}"
            for column in ("title", "body")
            for i in range(128)
        ),
    )
    assert output.columns[Stype.text] == ()
    torch.testing.assert_close(
        output.numerical, expected, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(output.categorical.code, table.categorical.code)
    torch.testing.assert_close(output.datetime, table.datetime)
    output.numerical.zero_()
    torch.testing.assert_close(table.numerical, original_numerical)
    assert table.text.to_arrow().to_pylist() == original_text
    torch.testing.assert_close(
        processor(table).numerical, expected, atol=1e-5, rtol=1e-5
    )


@withCUDA
def test_ragged_token_batches(device: torch.device) -> None:
    tokenizer = _CuDFTokenizer(
        vocabulary=cast(Any, None),
        normalizer=cast(Any, None),
        cls_token_id=101,
        sep_token_id=102,
        pad_token_id=0,
        max_length=5,
        max_input_chars_per_word=100,
    )
    flat_values = torch.tensor(
        [11, 12, 13, 14, 21], device=device, dtype=torch.int32
    )
    offsets = torch.tensor([0, 0, 4, 5], device=device, dtype=torch.int32)
    lengths = torch.tensor([2, 5, 3], device=device, dtype=torch.int32)
    batch = tokenizer.batch(
        flat_values=flat_values,
        offsets=offsets,
        lengths=lengths,
        indices=torch.tensor([2, 0, 1], device=device),
        max_length=5,
    )
    torch.testing.assert_close(
        batch["input_ids"],
        torch.tensor(
            [
                [101, 21, 102, 0, 0],
                [101, 102, 0, 0, 0],
                [101, 11, 12, 13, 102],
            ],
            device=device,
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        batch["attention_mask"],
        torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]],
            device=device,
            dtype=torch.int32,
        ),
    )
    batch = tokenizer.batch(
        flat_values=flat_values[:0],
        offsets=offsets[:2],
        lengths=lengths[:1],
        indices=torch.tensor([0], device=device),
        max_length=2,
    )
    torch.testing.assert_close(
        batch["input_ids"],
        torch.tensor([[101, 102]], device=device, dtype=torch.int32),
    )


@withCUDA
def test_encoder_preserves_batches_and_destinations_across_token_windows(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmbeddingModel(torch.nn.Module):
        def forward(
            self, features: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            ids = features["input_ids"]
            return {
                "sentence_embedding": torch.stack(
                    [
                        ids.sum(dim=1),
                        features["attention_mask"].sum(dim=1),
                        torch.full_like(ids[:, 0], ids.size(1)),
                    ],
                    dim=-1,
                ).float()
            }

    num_strings = 53
    counts = [(i * 7) % 11 for i in range(num_strings)]
    lengths = torch.tensor(counts, device=device, dtype=torch.int32).add_(2)
    offsets = torch.zeros(num_strings + 1, device=device, dtype=torch.int32)
    torch.cumsum(lengths.sub(2), dim=0, out=offsets[1:])
    values = torch.tensor(
        [i + 1 for i, count in enumerate(counts) for _ in range(count)],
        device=device,
        dtype=torch.int32,
    )
    tokenizer = _CuDFTokenizer(
        vocabulary=cast(Any, None),
        normalizer=cast(Any, None),
        cls_token_id=101,
        sep_token_id=102,
        pad_token_id=0,
        max_length=12,
        max_input_chars_per_word=100,
    )
    monkeypatch.setattr(
        tokenizer, "tokenize", lambda text: (values, offsets, lengths)
    )
    encoder = _Encoder(
        cast(Any, EmbeddingModel()), batch_size=3, embedding_dim=3
    )
    monkeypatch.setitem(encoder.__dict__, "_cudf_tokenizer", tokenizer)

    destination = torch.full(
        (num_strings, 8), -99, device=device, dtype=torch.float64
    )
    encoder._forward_cudf(
        text=StringTensor.from_list([""] * num_strings, device=device),
        out=destination[:, 2:].view(num_strings, 2, 3),
        indices=torch.arange(num_strings, device=device).mul_(2).add_(1),
    )

    expected = torch.full_like(destination, -99)
    expected[:, 5] = torch.tensor(
        [203 + (i + 1) * count for i, count in enumerate(counts)],
        device=device,
    )
    expected[:, 6] = lengths
    ordered_indices = lengths.argsort()
    for batch_indices in ordered_indices.split(3):
        expected[batch_indices, 7] = (
            lengths[batch_indices].max().to(expected.dtype)
        )
    torch.testing.assert_close(destination, expected)


@onlyCUDA
@pytest.mark.parametrize("num_numerical", [0, 2])
def test_gpu_matches_encode(num_numerical: int) -> None:
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
            numerical=torch.arange(
                len(texts) * num_numerical,
                dtype=torch.float64,
                device=device,
            ).reshape(len(texts), num_numerical),
            text=StringTensor.from_list(texts, device=device),
        )
        gpu_output = processor(table)
        gpu_emb = gpu_output.numerical[:, num_numerical:]
        torch.testing.assert_close(
            gpu_output.numerical[:, :num_numerical], table.numerical
        )

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

        torch.testing.assert_close(
            gpu_emb, ref_emb.to(gpu_emb.dtype), atol=1e-5, rtol=1e-5
        )


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
