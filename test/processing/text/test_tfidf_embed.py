import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing.text.tfidf_text_embed import TfidfTextEmbed
from sdm.testing import onlyCUDA


def _text_table(*columns: list[str]) -> TableTensor:
    names = tuple(f"t{i}" for i in range(len(columns)))
    rows = list(zip(*columns))
    return TableTensor(
        columns={"text": names},
        text=StringTensor.from_list([list(row) for row in rows]),
    )


def _numerical_by_ngram(
    encoder: TfidfTextEmbed,
    output: TableTensor,
) -> dict[str, torch.Tensor]:
    vocabulary = encoder._vocabularies[0].to_pylist()
    return {
        ngram: output.numerical[..., index].detach().cpu()
        for index, ngram in enumerate(vocabulary)
    }


def test_tfidf_encoder_outputs_numerical_block() -> None:
    table = TableTensor(
        columns={"text": ("t0", "t1")},
        text=StringTensor.from_list(
            [["hello world", "cat"], ["hello there", "dog"]]
        ),
    )

    output = TfidfTextEmbed(ngram_range=(2, 3)).fit_transform(table)

    assert output.columns[Stype.text] == ()
    names = output.columns[Stype.numerical]
    assert len(names) == output.numerical.size(-1)
    assert names[:32] == tuple(f"t0_{i}" for i in range(32))
    assert names[32:] == tuple(f"t1_{i}" for i in range(14))
    assert output.numerical.size(-1) == 46
    assert output.numerical.size(0) == 2
    assert output.numerical.dtype.is_floating_point


def test_tfidf_encoder_preserves_leading_dimensions() -> None:
    table = TableTensor(
        columns={"text": ("t0",)},
        text=StringTensor.from_list(
            [
                [["ab"], ["ac"]],
                [["ab"], ["bc"]],
            ],
        ),
    )

    output = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(table)

    assert output.numerical.shape[:-1] == (2, 2)
    assert output.columns[Stype.text] == ()


def test_tfidf_encoder_identical_strings_encode_identically() -> None:
    table = _text_table(["same text", "same text", "other"])

    output = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(table)

    assert torch.equal(output.numerical[0], output.numerical[1])
    assert not torch.equal(output.numerical[0], output.numerical[2])


def test_tfidf_encoder_rows_are_l2_normalized() -> None:
    table = _text_table(["short", "a much longer text cell"])

    output = TfidfTextEmbed(ngram_range=(2, 3)).fit_transform(table)

    assert torch.allclose(
        output.numerical.norm(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_tfidf_encoder_exact_values() -> None:
    table = _text_table(["ab", "ab ac", "ac"])
    encoder = TfidfTextEmbed(ngram_range=(2, 2))

    output = encoder.fit_transform(table)

    expected = {
        " a": torch.tensor([0.48133417, 0.61335554, 0.48133417]),
        "ab": torch.tensor([0.61980538, 0.39490346, 0.00000000]),
        "b ": torch.tensor([0.61980538, 0.39490346, 0.00000000]),
        "ac": torch.tensor([0.00000000, 0.39490346, 0.61980538]),
        "c ": torch.tensor([0.00000000, 0.39490346, 0.61980538]),
    }
    actual = _numerical_by_ngram(encoder, output)
    assert actual.keys() == expected.keys()
    for ngram, expected_values in expected.items():
        assert torch.allclose(actual[ngram], expected_values, atol=1e-6)


def test_tfidf_encoder_ignores_unseen_ngrams() -> None:
    encoder = TfidfTextEmbed(ngram_range=(3, 3))
    encoder.fit(_text_table(["aaa bbb", "aaa ccc"]))

    output = encoder.transform(_text_table(["zzz yyy", "aaa bbb"]))

    assert output.numerical[0].eq(0).all()
    assert output.numerical[1].ne(0).any()


def test_tfidf_encoder_max_features_keeps_most_frequent_ngrams() -> None:
    table = _text_table(["aa aa aa ab ab ac"])

    full = TfidfTextEmbed(ngram_range=(2, 3)).fit_transform(table)
    encoder = TfidfTextEmbed(ngram_range=(2, 3), max_features=3)
    capped = encoder.fit_transform(table)

    assert full.numerical.size(-1) > 3
    assert capped.numerical.size(-1) == 3
    assert set(encoder._vocabularies[0].to_pylist()) == {" a", "aa", "a "}


@pytest.mark.parametrize("max_features", [None, 3])
@onlyCUDA
def test_tfidf_encoder_cuda_fit_transform_matches_cpu(
    max_features: int | None,
) -> None:
    pytest.importorskip("cudf")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")

    texts = [["aa aa aa ab ab ac"], ["aa ab"], ["ac"]]
    cpu_table = TableTensor(
        columns={"text": ("t0",)},
        text=StringTensor.from_list(texts),
    )
    cuda_table = TableTensor(
        columns={"text": ("t0",)},
        text=StringTensor.from_list(texts, device="cuda"),
    )

    cpu_encoder = TfidfTextEmbed(
        ngram_range=(2, 3),
        max_features=max_features,
    )
    expected = cpu_encoder.fit_transform(cpu_table)
    cuda_encoder = TfidfTextEmbed(
        ngram_range=(2, 3),
        max_features=max_features,
    )
    output = cuda_encoder.fit_transform(cuda_table)

    expected_by_ngram = _numerical_by_ngram(cpu_encoder, expected)
    actual_by_ngram = _numerical_by_ngram(cuda_encoder, output)
    assert output.numerical.is_cuda
    assert actual_by_ngram.keys() == expected_by_ngram.keys()
    for ngram, expected_values in expected_by_ngram.items():
        assert torch.allclose(
            actual_by_ngram[ngram],
            expected_values,
            atol=1e-6,
        )


def test_tfidf_encoder_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfTextEmbed(ngram_range=(2, 2)).transform(_text_table(["a"]))


def test_tfidf_encoder_state_dict_round_trip(tmp_path) -> None:
    table = TableTensor(
        columns={"text": ("t0", "t1")},
        text=StringTensor.from_list(
            [["hello world", "cat"], ["hello there", "dog"]]
        ),
    )
    encoder = TfidfTextEmbed(ngram_range=(2, 3))
    expected = encoder.fit_transform(table)

    path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), path)

    restored = TfidfTextEmbed(ngram_range=(2, 3))
    restored.load_state_dict(torch.load(path, weights_only=True))

    assert torch.equal(restored.transform(table).numerical, expected.numerical)


def test_tfidf_encoder_unfitted_state_dict_round_trip() -> None:
    restored = TfidfTextEmbed(ngram_range=(2, 2))
    restored.load_state_dict(TfidfTextEmbed(ngram_range=(2, 2)).state_dict())

    with pytest.raises(RuntimeError, match="not fitted"):
        restored.transform(_text_table(["a"]))


def test_tfidf_encoder_load_state_dict_clears_stale_idf_buffers() -> None:
    restored = TfidfTextEmbed(ngram_range=(2, 2))
    restored.fit(_text_table(["hello world"], ["cat dog"]))

    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    table = _text_table(["hello world"])
    expected = encoder.fit_transform(table)
    restored.load_state_dict(encoder.state_dict(), strict=False)

    output = restored.transform(table)
    assert output.columns == expected.columns
    assert torch.equal(output.numerical, expected.numerical)


def test_tfidf_encoder_to_moves_fitted_state() -> None:
    table = _text_table(["hello world", "hello there"])
    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    encoder.fit(table)

    encoder.to(torch.float64)

    assert encoder.transform(table).numerical.dtype == torch.float64


def test_tfidf_encoder_failed_refit_preserves_previous_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _text_table(["hello world", "hello there"])
    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    expected = encoder.fit_transform(table)

    def fail_prune(*args: object, **kwargs: object) -> None:
        raise RuntimeError("refit failed")

    monkeypatch.setattr(encoder, "_prune", fail_prune)

    with pytest.raises(RuntimeError, match="refit failed"):
        encoder.fit(table)

    output = encoder.transform(table)
    assert output.columns == expected.columns
    assert torch.equal(output.numerical, expected.numerical)


def test_tfidf_encoder_refit_replaces_previous_state() -> None:
    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    encoder.fit(_text_table(["hello world"], ["cat dog"]))

    table = _text_table(["hello world"])
    expected = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(table)

    output = encoder.fit_transform(table)

    assert output.columns == expected.columns
    assert torch.equal(output.numerical, expected.numerical)
    assert sorted(encoder._buffers) == ["idf_0"]


def test_tfidf_encoder_empty_string_yields_zero_width_output() -> None:
    output = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(
        _text_table([""])
    )

    assert output.numerical.shape == (1, 0)
    assert output.columns[Stype.numerical] == ()


def test_tfidf_encoder_short_word_counts_once() -> None:
    output = TfidfTextEmbed(ngram_range=(5, 5)).fit_transform(
        _text_table(["a"])
    )

    assert output.numerical.shape == (1, 1)
    assert torch.equal(output.numerical, torch.ones(1, 1))


def test_tfidf_encoder_can_preserve_case() -> None:
    table = _text_table(["CAT", "cat"])

    lowercased = TfidfTextEmbed(ngram_range=(3, 3)).fit_transform(table)
    case_sensitive = TfidfTextEmbed(
        ngram_range=(3, 3),
        lowercase=False,
    ).fit_transform(table)

    assert torch.equal(lowercased.numerical[0], lowercased.numerical[1])
    assert not torch.equal(
        case_sensitive.numerical[0],
        case_sensitive.numerical[1],
    )
