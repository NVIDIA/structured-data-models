import copy

import pytest
import torch
from sdm import StringTensor, Stype, TableTensor
from sdm.processing.tfidf_encoder import TfidfEncoder
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

    output = TfidfEncoder(ngram_range=(2, 3)).fit_transform(table)

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

    output = TfidfEncoder(ngram_range=(2, 2)).fit_transform(table)

    assert torch.equal(output.numerical[0], output.numerical[1])
    assert not torch.equal(output.numerical[0], output.numerical[2])


def test_tfidf_encoder_rows_are_l2_normalized() -> None:
    table = _text_table(["short", "a much longer text cell"])

    output = TfidfEncoder(ngram_range=(2, 3)).fit_transform(table)

    assert torch.allclose(
        output.numerical.norm(dim=1),
        torch.ones(2),
        atol=1e-6,
    )


def test_tfidf_encoder_ignores_unseen_ngrams() -> None:
    encoder = TfidfEncoder(ngram_range=(3, 3))
    encoder.fit(_text_table(["aaa bbb", "aaa ccc"]))

    output = encoder.transform(_text_table(["zzz yyy", "aaa bbb"]))

    assert output.numerical[0].eq(0).all()
    assert output.numerical[1].ne(0).any()


def test_tfidf_encoder_max_features_caps_width() -> None:
    table = _text_table(["a bunch of different words here"])

    full = TfidfEncoder(ngram_range=(2, 3)).fit_transform(table)
    capped = TfidfEncoder(ngram_range=(2, 3), max_features=5).fit_transform(
        table
    )

    assert full.numerical.size(-1) > 5
    assert capped.numerical.size(-1) == 5


def test_tfidf_encoder_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TfidfEncoder(ngram_range=(2, 2)).transform(_text_table(["a"]))


def test_tfidf_encoder_shares_fitted_state_across_deepcopy() -> None:
    table = _text_table(["hello world", "hello there"])
    encoder = TfidfEncoder(ngram_range=(2, 2))
    encoder.fit(table)

    member = copy.deepcopy(encoder)

    assert member._state is encoder._state
    assert torch.equal(
        member.transform(table).numerical,
        encoder.transform(table).numerical,
    )


def test_tfidf_encoder_state_dict_round_trip(tmp_path) -> None:
    table = TableTensor(
        columns={"text": ("t0", "t1")},
        text=StringTensor.from_list(
            [["hello world", "cat"], ["hello there", "dog"]]
        ),
    )
    encoder = TfidfEncoder(ngram_range=(2, 3))
    expected = encoder.fit_transform(table)

    path = tmp_path / "encoder.pt"
    torch.save(encoder.state_dict(), path)

    restored = TfidfEncoder(ngram_range=(2, 3))
    restored.load_state_dict(torch.load(path, weights_only=True))

    assert torch.equal(restored.transform(table).numerical, expected.numerical)


def test_tfidf_encoder_unfitted_state_dict_round_trip() -> None:
    restored = TfidfEncoder(ngram_range=(2, 2))
    restored.load_state_dict(TfidfEncoder(ngram_range=(2, 2)).state_dict())

    with pytest.raises(RuntimeError, match="not fitted"):
        restored.transform(_text_table(["a"]))


def test_tfidf_encoder_to_moves_fitted_state() -> None:
    table = _text_table(["hello world", "hello there"])
    encoder = TfidfEncoder(ngram_range=(2, 2))
    encoder.fit(table)

    encoder.to(torch.float64)

    assert encoder.transform(table).numerical.dtype == torch.float64


@onlyCUDA
def test_tfidf_encoder_to_moves_fitted_state_cuda() -> None:
    texts = ["hello world", "hello there"]
    encoder = TfidfEncoder(ngram_range=(2, 2))
    encoder.fit(_text_table(texts))
    query = TableTensor(
        columns={"text": ("t0",)},
        text=StringTensor.from_list([[t] for t in texts], device="cuda"),
    )

    encoder.to("cuda")

    output = encoder.transform(query)
    assert output.numerical.is_cuda
