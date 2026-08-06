from __future__ import annotations

import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EmbedText


def _fake_embed(strings: list[str]) -> Tensor:
    return torch.ones(len(strings), 2)


def test_forward() -> None:
    table = TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["a", "b"],
                ["c", "d"],
            ]
        ),
    )

    output = EmbedText(
        _fake_embed,
        embedding_dim=2,
    )(table)

    assert output.columns[Stype.numerical] == (
        "title_0",
        "title_1",
        "body_0",
        "body_1",
    )
    assert torch.equal(output.numerical, torch.ones(2, 4))
