import copy
from typing import cast

import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EmbedText, StypeDispatch


class TextLength(torch.nn.Module):
    def forward(self, text: StringTensor) -> torch.Tensor:
        _, offset = cast(StringTensor, text.contiguous()).data_offset
        length = offset.diff().reshape(text.size())
        return torch.stack((length, length.square()), dim=-1).float()


def _table() -> TableTensor:
    return TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["a", "hello"],
                ["cat", ""],
            ]
        ),
    )


def test_embed_text() -> None:
    output = EmbedText(TextLength()).transform(_table())

    assert output.columns[Stype.text] == ()
    assert output.columns[Stype.numerical] == (
        "title__0",
        "title__1",
        "body__0",
        "body__1",
    )
    assert output.numerical.equal(
        torch.tensor(
            [
                [1.0, 1.0, 5.0, 25.0],
                [3.0, 9.0, 0.0, 0.0],
            ]
        )
    )


def test_embed_text_composes_with_stype_dispatch() -> None:
    table = _table()
    processor = StypeDispatch(
        text=EmbedText(TextLength()),
        remainder="passthrough",
    )

    context = processor.fit_transform(table[:1])
    query = processor.transform(table[1:])

    assert context.schema == query.schema
    assert context.size() == (1, 4)
    assert query.size() == (1, 4)


def test_embed_text_registers_encoder() -> None:
    encoder = torch.nn.Linear(2, 2)
    processor = EmbedText(encoder)
    processor.eval()
    copied = copy.deepcopy(processor)

    assert processor.encoder is encoder
    assert copied is not processor
    assert copied.encoder is encoder
    assert not processor.training
    assert not copied.training
    assert not copied.encoder.training
    assert tuple(processor.parameters()) == tuple(encoder.parameters())


@pytest.mark.parametrize(
    "output",
    [
        torch.ones(2),
        torch.ones(2, 1, dtype=torch.int64),
        torch.ones(2, 2, 1),
        torch.ones(2, 0),
        "not a tensor",
    ],
)
def test_embed_text_rejects_invalid_output(output: object) -> None:
    class InvalidEncoder(torch.nn.Module):
        def forward(self, text: StringTensor) -> object:
            return output

    table = TableTensor(
        columns={"text": ("review",)},
        text=StringTensor.from_list([["good"], ["bad"]]),
    )

    with pytest.raises(ValueError, match=r"encoder.*shape"):
        EmbedText(InvalidEncoder()).transform(table)
