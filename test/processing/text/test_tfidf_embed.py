import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing.text.tfidf_text_embed import TfidfTextEmbed
from sdm.testing import onlyCUDA


def _numerical_by_column_ngram(
    encoder: TfidfTextEmbed,
    output: TableTensor,
) -> dict[tuple[int, str], torch.Tensor]:
    values: dict[tuple[int, str], torch.Tensor] = {}
    offset = 0
    for column, vocabulary in enumerate(encoder._vocabularies):
        for index, ngram in enumerate(vocabulary.to_pylist()):
            values[(column, ngram)] = (
                output.numerical[..., offset + index].detach().cpu()
            )
        offset += len(vocabulary)
    return values


def test_tfidf_encoder_preserves_leading_dimensions() -> None:
    table = TableTensor.from_tensor(
        StringTensor.from_list(
            [
                [["ab"], ["ac"]],
                [["ab"], ["bc"]],
            ]
        )
    )

    output = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(table)

    assert output.numerical.shape[:-1] == (2, 2)


def test_tfidf_encoder_rows_are_l2_normalized() -> None:
    table = TableTensor.from_tensor(
        StringTensor.from_list([["short"], ["a much longer text cell"]])
    )

    output = TfidfTextEmbed(ngram_range=(2, 3)).fit_transform(table)

    assert torch.allclose(
        output.numerical.norm(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_tfidf_encoder_exact_values() -> None:
    table = TableTensor.from_tensor(
        StringTensor.from_list([["ab"], ["ab ac"], ["ac"]])
    )
    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    output = encoder.fit_transform(table)

    expected = {
        (0, " a"): torch.tensor([0.48133417, 0.61335554, 0.48133417]),
        (0, "ab"): torch.tensor([0.61980538, 0.39490346, 0.00000000]),
        (0, "b "): torch.tensor([0.61980538, 0.39490346, 0.00000000]),
        (0, "ac"): torch.tensor([0.00000000, 0.39490346, 0.61980538]),
        (0, "c "): torch.tensor([0.00000000, 0.39490346, 0.61980538]),
    }
    actual = _numerical_by_column_ngram(encoder, output)

    assert actual.keys() == expected.keys()
    for key, expected_values in expected.items():
        assert torch.allclose(actual[key], expected_values, atol=1e-6)


def test_tfidf_encoder_ignores_unseen_ngrams() -> None:
    train = TableTensor.from_tensor(
        StringTensor.from_list([["aaa bbb"], ["aaa ccc"]])
    )
    query = TableTensor.from_tensor(
        StringTensor.from_list([["zzz yyy"], ["aaa bbb"]])
    )
    encoder = TfidfTextEmbed(ngram_range=(3, 3))
    encoder.fit(train)

    output = encoder.transform(query)

    assert output.numerical[0].eq(0).all()
    assert output.numerical[1].ne(0).any()


def test_tfidf_encoder_max_features_keeps_most_frequent_ngrams() -> None:
    table = TableTensor.from_tensor(
        StringTensor.from_list([["aa aa aa ab ab ac"]])
    )
    full = TfidfTextEmbed(ngram_range=(2, 3)).fit_transform(table)
    encoder = TfidfTextEmbed(ngram_range=(2, 3), max_features=3)
    capped = encoder.fit_transform(table)

    assert full.numerical.size(-1) > 3
    assert capped.numerical.size(-1) == 3
    assert set(encoder._vocabularies[0].to_pylist()) == {" a", "aa", "a "}


def test_tfidf_encoder_empty_string_yields_zero_width_output() -> None:
    table = TableTensor.from_tensor(StringTensor.from_list([[""]]))

    output = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(table)

    assert output.numerical.shape == (1, 0)
    assert output.columns[Stype.numerical] == ()


def test_tfidf_encoder_null_text_yields_zero_features() -> None:
    table = TableTensor.from_tensor(StringTensor.from_list([[None], ["ab"]]))

    output = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(table)

    assert output.numerical.size(0) == 2
    assert output.numerical[0].eq(0).all()
    assert output.numerical[1].ne(0).any()


def test_tfidf_encoder_short_word_counts_once() -> None:
    table = TableTensor.from_tensor(StringTensor.from_list([["a"]]))

    output = TfidfTextEmbed(ngram_range=(5, 5)).fit_transform(table)

    assert output.numerical.shape == (1, 1)
    assert torch.equal(output.numerical, torch.ones(1, 1))


def test_tfidf_encoder_can_preserve_case() -> None:
    table = TableTensor.from_tensor(StringTensor.from_list([["CAT"], ["cat"]]))

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


@pytest.mark.parametrize(
    ("train", "query", "ngram_range", "max_features", "lowercase"),
    [
        pytest.param(
            [["aa aa aa ab ab ac"], ["aa ab"], ["ac"]],
            None,
            (2, 3),
            None,
            True,
            id="basic",
        ),
        pytest.param(
            [["aa aa aa ab ab ac"], ["aa ab"], ["ac"]],
            None,
            (2, 3),
            3,
            True,
            id="max-features",
        ),
        pytest.param(
            [[""]],
            None,
            (2, 2),
            None,
            True,
            id="empty-string",
        ),
        pytest.param(
            [["a"]],
            None,
            (5, 5),
            None,
            True,
            id="short-word",
        ),
        pytest.param(
            [["aaa bbb"], ["aaa ccc"]],
            [["zzz yyy"], ["aaa bbb"]],
            (3, 3),
            None,
            True,
            id="unseen-ngrams",
        ),
        pytest.param(
            [["CAT"], ["cat"]],
            None,
            (3, 3),
            None,
            False,
            id="case-sensitive",
        ),
        pytest.param(
            [["hello world", "cat"], ["hello there", "dog"]],
            None,
            (2, 3),
            None,
            True,
            id="multi-column",
        ),
        pytest.param(
            [[None], ["ab"]],
            None,
            (2, 2),
            None,
            True,
            id="null-text",
        ),
    ],
)
@onlyCUDA
def test_tfidf_encoder_cuda_matches_cpu(
    train: list[list[str | None]],
    query: list[list[str | None]] | None,
    ngram_range: tuple[int, int],
    max_features: int | None,
    lowercase: bool,
) -> None:
    pytest.importorskip("cudf")
    pytest.importorskip("cupy")
    pytest.importorskip("pylibcudf")

    cpu_train = TableTensor.from_tensor(StringTensor.from_list(train))
    cuda_train = TableTensor.from_tensor(
        StringTensor.from_list(train, device="cuda")
    )
    cpu_encoder = TfidfTextEmbed(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=lowercase,
    )
    cuda_encoder = TfidfTextEmbed(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=lowercase,
    )

    if query is None:
        expected = cpu_encoder.fit_transform(cpu_train)
        output = cuda_encoder.fit_transform(cuda_train)
    else:
        cpu_encoder.fit(cpu_train)
        cuda_encoder.fit(cuda_train)
        expected = cpu_encoder.transform(
            TableTensor.from_tensor(StringTensor.from_list(query))
        )
        output = cuda_encoder.transform(
            TableTensor.from_tensor(
                StringTensor.from_list(query, device="cuda")
            )
        )

    expected_by_ngram = _numerical_by_column_ngram(cpu_encoder, expected)
    actual_by_ngram = _numerical_by_column_ngram(cuda_encoder, output)
    assert output.numerical.is_cuda
    assert output.numerical.shape == expected.numerical.shape
    assert output.columns == expected.columns
    assert actual_by_ngram.keys() == expected_by_ngram.keys()
    for key, expected_values in expected_by_ngram.items():
        assert torch.allclose(
            actual_by_ngram[key],
            expected_values,
            atol=1e-6,
        )


def test_tfidf_encoder_state_dict_round_trip(tmp_path) -> None:
    table = TableTensor.from_tensor(
        StringTensor.from_list(
            [["hello world", "cat"], ["hello there", "dog"]]
        )
    )
    encoder = TfidfTextEmbed(ngram_range=(2, 3))
    expected = encoder.fit_transform(table)

    path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), path)
    restored = TfidfTextEmbed(ngram_range=(2, 3))
    restored.load_state_dict(torch.load(path, weights_only=True))

    assert torch.equal(restored.transform(table).numerical, expected.numerical)


def test_tfidf_encoder_load_state_dict_clears_stale_idf_buffers() -> None:
    wide = TableTensor.from_tensor(
        StringTensor.from_list([["hello world", "cat dog"]])
    )
    narrow = TableTensor.from_tensor(StringTensor.from_list([["hello world"]]))
    restored = TfidfTextEmbed(ngram_range=(2, 2))
    restored.fit(wide)

    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    expected = encoder.fit_transform(narrow)
    restored.load_state_dict(encoder.state_dict(), strict=False)

    output = restored.transform(narrow)
    assert output.columns == expected.columns
    assert torch.equal(output.numerical, expected.numerical)


def test_tfidf_encoder_failed_refit_preserves_previous_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = TableTensor.from_tensor(
        StringTensor.from_list([["hello world"], ["hello there"]])
    )
    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    expected = encoder.fit_transform(table)

    def fail_bincount(*args: object, **kwargs: object) -> None:
        raise RuntimeError("refit failed")

    monkeypatch.setattr(torch, "bincount", fail_bincount)

    with pytest.raises(RuntimeError, match="refit failed"):
        encoder.fit(table)

    output = encoder.transform(table)
    assert output.columns == expected.columns
    assert torch.equal(output.numerical, expected.numerical)


def test_tfidf_encoder_refit_replaces_previous_state() -> None:
    wide = TableTensor.from_tensor(
        StringTensor.from_list([["hello world", "cat dog"]])
    )
    narrow = TableTensor.from_tensor(StringTensor.from_list([["hello world"]]))
    encoder = TfidfTextEmbed(ngram_range=(2, 2))
    encoder.fit(wide)
    expected = TfidfTextEmbed(ngram_range=(2, 2)).fit_transform(narrow)

    output = encoder.fit_transform(narrow)

    assert output.columns == expected.columns
    assert torch.equal(output.numerical, expected.numerical)
