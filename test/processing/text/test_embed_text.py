from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, ClassVar

import pytest
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EmbedText


class _FakeSentenceTransformer(torch.nn.Module):
    instances: ClassVar[list[_FakeSentenceTransformer]] = []

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.encode_kwargs: dict[str, Any] = {}
        self.register_buffer("weight", torch.ones(1))
        self.instances.append(self)

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, strings: list[str], **kwargs: Any) -> Tensor:
        self.strings = strings
        self.encode_kwargs = kwargs
        return torch.arange(len(strings) * 2).reshape(len(strings), 2)


@pytest.fixture(autouse=True)
def sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module: Any = ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeSentenceTransformer.instances.clear()


def test_forward() -> None:
    table = TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["a", "b"],
                ["c", None],
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
    assert torch.equal(
        output.numerical,
        torch.tensor(
            [
                [0.0, 1.0, 4.0, 5.0],
                [2.0, 3.0, 6.0, 7.0],
            ]
        ),
    )
    model = _FakeSentenceTransformer.instances[0]
    assert model.model_name == "fake-model"
    assert model.strings == ["a", "c", "b", ""]
    assert model.encode_kwargs == {
        "show_progress_bar": False,
        "convert_to_tensor": True,
        "device": "cpu",
    }


def test_to_moves_model() -> None:
    processor = EmbedText("fake-model")

    processor.to(dtype=torch.float64)

    model = _FakeSentenceTransformer.instances[0]
    assert model.weight.dtype == torch.float64
