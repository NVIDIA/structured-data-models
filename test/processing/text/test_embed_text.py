from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, ClassVar

import pytest
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EmbedText
from sdm.testing import withCUDA


class MyModel(torch.nn.Module):
    instances: ClassVar[list[MyModel]] = []

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.encode_kwargs: dict[str, Any] = {}
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
    module.SentenceTransformer = MyModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    MyModel.instances.clear()


@withCUDA
def test_forward(device: torch.device) -> None:
    table = TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["a", "b"],
                ["c", None],
            ],
            device=device,
        ),
    )

    output = EmbedText("fake-model").to(device)(table)

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
            ],
            device=device,
        ),
    )
    model = MyModel.instances[0]
    assert model.model_name == "fake-model"
    assert model.strings == ["a", "c", "b", ""]
    assert model.encode_kwargs == {
        "show_progress_bar": False,
        "convert_to_tensor": True,
        "device": str(device),
    }
