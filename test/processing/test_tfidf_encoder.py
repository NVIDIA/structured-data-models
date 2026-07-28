import pytest
import torch
from sdm import StringTensor, Stype, TableTensor
from sdm.processing.text.tfidf_transformer import TfidfTransformer
from sdm.testing import onlyCUDA


def _text_table(*columns: list[str]) -> TableTensor:
    names = tuple(f"t{i}" for i in range(len(columns)))
    rows = list(zip(*columns))
    return TableTensor(
        columns={"text": names},
        text=StringTensor.from_list([list(row) for row in rows]),
    )


def test_tfidf_encoder_outputs_numerical_block() -> None:
    table = TableTensor(
        columns={"text": ("t0", "t1")},
        text=StringTensor.from_list(
            [["hello world", "cat"], ["hello there", "dog"]]
        ),
    )

    output = TfidfTransformer(ngram_range=(2, 3)).fit_transform(table)

    assert output.columns[Stype.text] == ()
    names = output.columns[Stype.numerical]
    assert len(names) == output.numerical.size(-1)
    assert names[:32] == tuple(f"t0_{i}" for i in range(32))
    assert names[32:] == tuple(f"t1_{i}" for i in range(14))
    assert output.numerical.size(-1) == 46
    assert output.numerical.size(0) == 2
    assert output.numerical.dtype.is_floating_point


def test_tfidf_encoder_identical_strings_encode_identically() -> None:
    table = _text_table(["same text", "same text", "other"])

    output = TfidfTransformer(ngram_range=(2, 2)).fit_transform(table)

    assert torch.equal(output.numerical[0], output.numerical[1])
    assert not torch.equal(output.numerical[0], output.numerical[2])


def test_tfidf_encoder_rows_are_l2_normalized() -> None:
    table = _text_table(["short", "a much longer text cell"])

    output = TfidfTransformer(ngram_range=(2, 3)).fit_transform(table)

    assert torch.allclose(
        output.numerical.norm(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_tfidf_encoder_exact_values() -> None:
    # Bigram counts per document, each word padded to " ab ":
    #   d0 "ab"     -> {" a": 1, "ab": 1, "b ": 1}
    #   d1 "ab ac"  -> {" a": 2, "ab": 1, "b ": 1, "ac": 1, "c ": 1}
    #   d2 "ac"     -> {" a": 1, "ac": 1, "c ": 1}
    # With n_docs = 3, df(" a") = 3 and df = 2 for every other n-gram, so
    # idf = ln((1 + n_docs) / (1 + df)) + 1 gives idf(" a") = 1.0 and
    # idf = ln(4 / 3) + 1 = 1.28768207 elsewhere. Each row is the raw
    # n-gram count times idf, L2-normalized.
    table = _text_table(["ab", "ab ac", "ac"])
    encoder = TfidfTransformer(ngram_range=(2, 2))

    output = encoder.fit_transform(table)

    expected = {  # per n-gram, the value for d0, d1, d2
        " a": [0.48133417, 0.61335554, 0.48133417],
        "ab": [0.61980538, 0.39490346, 0.00000000],
        "b ": [0.61980538, 0.39490346, 0.00000000],
        "ac": [0.00000000, 0.39490346, 0.61980538],
        "c ": [0.00000000, 0.39490346, 0.61980538],
    }
    # Vocabulary order differs between the CPU and CUDA factorization, so
    # line the expected columns up with the fitted vocabulary.
    vocabulary = encoder._vocabularies[0].to_pylist()
    assert sorted(vocabulary) == sorted(expected)
    assert torch.allclose(
        output.numerical,
        torch.tensor([expected[ngram] for ngram in vocabulary]).T,
        atol=1e-6,
    )


def test_tfidf_encoder_ignores_unseen_ngrams() -> None:
    encoder = TfidfTransformer(ngram_range=(3, 3))
    encoder.fit(_text_table(["aaa bbb", "aaa ccc"]))

    output = encoder.transform(_text_table(["zzz yyy", "aaa bbb"]))

    assert output.numerical[0].eq(0).all()
    assert output.numerical[1].ne(0).any()


def test_tfidf_encoder_max_features_caps_width() -> None:
    table = _text_table(["a bunch of different words here"])

    full = TfidfTransformer(ngram_range=(2, 3)).fit_transform(table)
    capped = TfidfTransformer(
        ngram_range=(2, 3), max_features=5
    ).fit_transform(table)

    assert full.numerical.size(-1) > 5
    assert capped.numerical.size(-1) == 5


def test_tfidf_encoder_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfTransformer(ngram_range=(2, 2)).transform(_text_table(["a"]))


def test_tfidf_encoder_state_dict_round_trip(tmp_path) -> None:
    table = TableTensor(
        columns={"text": ("t0", "t1")},
        text=StringTensor.from_list(
            [["hello world", "cat"], ["hello there", "dog"]]
        ),
    )
    encoder = TfidfTransformer(ngram_range=(2, 3))
    expected = encoder.fit_transform(table)

    path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), path)

    restored = TfidfTransformer(ngram_range=(2, 3))
    restored.load_state_dict(torch.load(path, weights_only=True))

    assert torch.equal(restored.transform(table).numerical, expected.numerical)


def test_tfidf_encoder_unfitted_state_dict_round_trip() -> None:
    restored = TfidfTransformer(ngram_range=(2, 2))
    restored.load_state_dict(TfidfTransformer(ngram_range=(2, 2)).state_dict())

    with pytest.raises(RuntimeError, match="not fitted"):
        restored.transform(_text_table(["a"]))


def test_tfidf_encoder_to_moves_fitted_state() -> None:
    table = _text_table(["hello world", "hello there"])
    encoder = TfidfTransformer(ngram_range=(2, 2))
    encoder.fit(table)

    encoder.to(torch.float64)

    assert encoder.transform(table).numerical.dtype == torch.float64


@onlyCUDA
def test_tfidf_encoder_to_moves_fitted_state_cuda() -> None:
    texts = ["hello world", "hello there"]
    encoder = TfidfTransformer(ngram_range=(2, 2))
    encoder.fit(_text_table(texts))
    query = TableTensor(
        columns={"text": ("t0",)},
        text=StringTensor.from_list([[t] for t in texts], device="cuda"),
    )

    encoder.to("cuda")

    output = encoder.transform(query)
    assert output.numerical.is_cuda


def _ngrams(
    values: list[str],
    ngram_range: tuple[int, int],
    *,
    lowercase: bool = True,
) -> tuple[StringTensor, torch.Tensor]:
    encoder = TfidfTransformer(ngram_range=ngram_range)
    tensor = StringTensor.from_list(values)
    return encoder._character_ngrams(tensor, ngram_range, lowercase=lowercase)


def test_character_ngrams() -> None:
    flat, offset = _ngrams(["cat", "hi cat"], (2, 2))

    assert flat.tolist() == [
        " c",
        "ca",
        "at",
        "t ",
        " h",
        "hi",
        "i ",
        " c",
        "ca",
        "at",
        "t ",
    ]
    assert offset.equal(torch.tensor([0, 4, 11]))


def test_character_ngrams_combines_sizes() -> None:
    flat, offset = _ngrams(["cat"], (2, 3))

    assert flat.tolist() == [
        " c",
        "ca",
        "at",
        "t ",
        " ca",
        "cat",
        "at ",
    ]
    assert offset.equal(torch.tensor([0, 7]))


def test_character_ngrams_short_word_counts_once() -> None:
    flat, offset = _ngrams(["a"], (5, 5))

    assert flat.tolist() == [" a "]
    assert offset.equal(torch.tensor([0, 1]))


def test_character_ngrams_empty_string_yields_nothing() -> None:
    flat, offset = _ngrams([""], (2, 2))

    assert flat.tolist() == []
    assert offset.equal(torch.tensor([0, 0]))


def test_character_ngrams_lowercases_by_default() -> None:
    flat, _ = _ngrams(["CAT"], (3, 3))
    assert flat.tolist() == [" ca", "cat", "at "]

    flat, _ = _ngrams(["CAT"], (3, 3), lowercase=False)
    assert flat.tolist() == [" CA", "CAT", "AT "]


def test_character_ngrams_rejects_multi_dimensional_input() -> None:
    encoder = TfidfTransformer(ngram_range=(2, 2))
    tensor = StringTensor.from_list([["a", "b"], ["c", "d"]])

    with pytest.raises(NotImplementedError, match="one-dimensional"):
        encoder._character_ngrams(tensor, (2, 2))
