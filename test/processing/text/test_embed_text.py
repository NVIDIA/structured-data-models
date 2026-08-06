from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EmbedText


def _make_fake_st(_model_name: str) -> MagicMock:
    model = MagicMock()
    model.get_embedding_dimension.return_value = 2
    model.encode.side_effect = lambda strings, **_kw: torch.ones(
        len(strings), 2
    )
    return model


def test_forward() -> None:
    fake_mod = ModuleType("sentence_transformers")
    setattr(fake_mod, "SentenceTransformer", _make_fake_st)
    sys.modules["sentence_transformers"] = fake_mod

    try:
        table = TableTensor(
            columns={"text": ("title", "body")},
            text=StringTensor.from_list(
                [
                    ["a", "b"],
                    ["c", "d"],
                ]
            ),
        )

        output = EmbedText("fake-model")(table)

        assert output.columns[Stype.numerical] == (
            "title_0",
            "title_1",
            "body_0",
            "body_1",
        )
        assert torch.equal(output.numerical, torch.ones(2, 4))
    finally:
        sys.modules.pop("sentence_transformers", None)
