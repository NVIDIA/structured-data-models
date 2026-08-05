from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EmbedText, Recipe, StypeDispatch
from sdm.testing import onlyCUDA

if TYPE_CHECKING:
    import cudf


class _FakeEmbeddingModel(torch.nn.Module):
    """Deterministic model: [len(s), len(s) ** 2] per string."""

    def forward(self, strings: pa.Array) -> Tensor:
        lengths = pc.call_function("utf8_length", [strings])
        values = torch.tensor(lengths.to_numpy(zero_copy_only=False))
        values = values.to(torch.get_default_dtype())
        return torch.stack((values, values.square()), dim=-1)


class _CudfEmbeddingModel(torch.nn.Module):
    def forward(self, strings: cudf.Series) -> Tensor:
        lengths = torch.from_dlpack(
            strings.str.len().astype("float32").to_dlpack()
        )
        return torch.stack((lengths, lengths.square()), dim=-1)


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


def test_embeds_each_text_column() -> None:
    output = EmbedText(
        _FakeEmbeddingModel(),
        embedding_dim=2,
    ).transform(_text_table())

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


def test_empty_rows_use_embedding_dim_without_model_call() -> None:
    table = TableTensor(
        columns={"text": ("title",)},
        text=StringTensor(
            data=torch.empty(0, dtype=torch.uint8),
            offset=torch.zeros(1, dtype=torch.int32),
            valid=None,
            size=(0, 1),
        ),
    )

    output = EmbedText(
        _FakeEmbeddingModel(),
        embedding_dim=2,
    ).transform(table)

    assert output.numerical.size() == (0, 2)
    assert output.columns[Stype.numerical] == ("title_0", "title_1")


@onlyCUDA
def test_uses_cudf_for_cuda_text() -> None:
    pytest.importorskip("cudf")
    table = TableTensor(
        columns={"text": ("title",)},
        text=StringTensor.from_list(
            [["ab"], ["abcd"]],
            device="cuda",
        ),
    )

    output = EmbedText(
        _CudfEmbeddingModel(),
        embedding_dim=2,
    ).transform(table)

    assert output.numerical.device.type == "cuda"
    assert output.columns[Stype.numerical] == ("title_0", "title_1")
    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[2.0, 4.0], [4.0, 16.0]], device="cuda"),
    )


def test_in_stype_dispatch_route() -> None:
    dispatch = StypeDispatch(
        text=EmbedText(
            _FakeEmbeddingModel(),
            embedding_dim=2,
        )
    )

    output = dispatch.fit_transform(_text_table())

    assert output.numerical.size() == (2, 4)
    assert output.columns[Stype.text] == ()


def test_deepcopy_shares_only_embedding_model() -> None:
    embedding_model = torch.nn.Linear(2, 2)
    processor = EmbedText(
        embedding_model,
        embedding_dim=2,
    )
    processor.register_buffer("state", torch.tensor([1.0]))
    recipe = Recipe(
        features=processor,
    )

    copied = copy.deepcopy(recipe)

    assert copied is not recipe
    assert copied.features is not recipe.features
    assert isinstance(copied.features, EmbedText)
    original_reference = processor.get_submodule("_embedding_model")
    copied_reference = copied.features.get_submodule("_embedding_model")
    assert copied_reference is not original_reference
    assert copied_reference.get_submodule("module") is embedding_model
    original_state = processor.get_buffer("state")
    copied_state = copied.features.get_buffer("state")
    assert copied_state is not original_state
    assert torch.equal(copied_state, original_state)
