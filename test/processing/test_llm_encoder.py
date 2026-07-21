import copy

import pytest
import torch
from sdm import StringTensor, Stype, TableTensor
from sdm.processing import Embedder, LLMEncoder, StypeDispatch
from torch import Tensor


class _FakeEmbedder:
    """Deterministic embedder: [len(s), len(s) ** 2] per string."""

    dim = 2

    def encode(self, strings: list[str]) -> Tensor:
        return torch.tensor(
            [[float(len(s)), float(len(s)) ** 2] for s in strings]
        )


class _WrongShapeEmbedder:
    dim = 2

    def encode(self, strings: list[str]) -> Tensor:
        return torch.zeros(len(strings) + 1, 2)


def _text_table() -> TableTensor:
    return TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["ab", "abcd"],
                ["a", "abc"],
            ]
        ),
    )


def test_llm_encoder_embeds_each_text_column() -> None:
    output = LLMEncoder(_FakeEmbedder()).transform(_text_table())

    assert output.columns[Stype.numerical] == (
        "title_0",
        "title_1",
        "body_0",
        "body_1",
    )
    assert output.columns[Stype.text] == ()
    assert torch.equal(
        output.numerical,
        torch.tensor(
            [
                [2.0, 4.0, 4.0, 16.0],
                [1.0, 1.0, 3.0, 9.0],
            ]
        ),
    )


def test_llm_encoder_requires_no_fit() -> None:
    encoder = LLMEncoder(_FakeEmbedder())

    assert encoder.requires_fit is False
    encoder.transform(_text_table())  # No prior `fit` call.


def test_llm_encoder_rejects_wrong_encode_shape() -> None:
    with pytest.raises(ValueError, match="dim"):
        LLMEncoder(_WrongShapeEmbedder()).transform(_text_table())


def test_llm_encoder_empty_rows_use_dim_without_encode() -> None:
    table = TableTensor(
        columns={"text": ("title",)},
        text=StringTensor(
            data=torch.empty(0, dtype=torch.uint8),
            offset=torch.zeros(1, dtype=torch.int32),
            size=(0, 1),
        ),
    )

    output = LLMEncoder(_FakeEmbedder()).transform(table)

    assert output.numerical.size() == (0, 2)
    assert output.columns[Stype.numerical] == ("title_0", "title_1")


def test_llm_encoder_shares_embedder_across_deepcopy() -> None:
    encoder = LLMEncoder(_FakeEmbedder())

    member = copy.deepcopy(encoder)

    assert member.embedder is encoder.embedder
    assert torch.equal(
        member.transform(_text_table()).numerical,
        encoder.transform(_text_table()).numerical,
    )


def test_llm_encoder_satisfies_embedder_protocol() -> None:
    assert isinstance(_FakeEmbedder(), Embedder)


def test_llm_encoder_in_stype_dispatch_route() -> None:
    dispatch = StypeDispatch(text=LLMEncoder(_FakeEmbedder()))

    output = dispatch.fit_transform(_text_table())

    assert output.numerical.size() == (2, 4)
    assert output.columns[Stype.text] == ()
