import copy

import pytest
import torch
from sdm import StringTensor, Stype, TableTensor
from sdm.processing import StypeDispatch, TfidfEncoder


def _text_table(*columns: list[str]) -> TableTensor:
    names = tuple(f"t{i}" for i in range(len(columns)))
    rows = list(zip(*columns))
    return TableTensor(
        columns={"text": names},
        text=StringTensor.from_list([list(row) for row in rows]),
    )


def test_tfidf_encoder_outputs_numerical_block() -> None:
    table = _text_table(["hello world", "hello there"], ["cat", "dog"])

    output = TfidfEncoder(ngram_range=(2, 3)).fit_transform(table)

    assert output.columns[Stype.text] == ()
    names = output.columns[Stype.numerical]
    assert len(names) == output.numerical.size(-1)
    assert names[0].startswith("t0_")
    assert names[-1].startswith("t1_")
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


def test_tfidf_encoder_in_stype_dispatch_route() -> None:
    table = _text_table(["hello world", "hello there"])
    dispatch = StypeDispatch(text=TfidfEncoder(ngram_range=(2, 2)))

    output = dispatch.fit_transform(table)

    assert output.columns[Stype.text] == ()
    assert output.numerical.size(0) == 2
    assert output.numerical.size(-1) > 0
