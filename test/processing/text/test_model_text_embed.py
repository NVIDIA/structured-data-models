from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import Recipe, StypeDispatch
from sdm.processing.text.model_text_embed import ModelTextEmbed
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


class _WrongShapeEmbeddingModel(torch.nn.Module):
    def forward(self, strings: pa.Array) -> Tensor:
        return torch.zeros(len(strings) + 1, 2)


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


def test_llm_text_embed_embeds_each_text_column() -> None:
    output = ModelTextEmbed(
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


def test_llm_text_embed_rejects_wrong_model_shape() -> None:
    with pytest.raises(ValueError, match="Expected 'embedding_model'"):
        ModelTextEmbed(
            _WrongShapeEmbeddingModel(),
            embedding_dim=2,
        ).transform(_text_table())


def test_llm_text_embed_empty_rows_use_embedding_dim_without_model_call() -> (
    None
):
    table = TableTensor(
        columns={"text": ("title",)},
        text=StringTensor(
            data=torch.empty(0, dtype=torch.uint8),
            offset=torch.zeros(1, dtype=torch.int32),
            valid=None,
            size=(0, 1),
        ),
    )

    output = ModelTextEmbed(
        _FakeEmbeddingModel(),
        embedding_dim=2,
    ).transform(table)

    assert output.numerical.size() == (0, 2)
    assert output.columns[Stype.numerical] == ("title_0", "title_1")


@onlyCUDA
def test_llm_text_embed_uses_cudf_for_cuda_text() -> None:
    pytest.importorskip("cudf")
    table = TableTensor(
        columns={"text": ("title",)},
        text=StringTensor.from_list(
            [["ab"], ["abcd"]],
            device="cuda",
        ),
    )

    output = ModelTextEmbed(
        _CudfEmbeddingModel(),
        embedding_dim=2,
    ).transform(table)

    assert output.numerical.device.type == "cuda"
    assert output.columns[Stype.numerical] == ("title_0", "title_1")
    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[2.0, 4.0], [4.0, 16.0]], device="cuda"),
    )


def test_llm_text_embed_in_stype_dispatch_route() -> None:
    dispatch = StypeDispatch(
        text=ModelTextEmbed(
            _FakeEmbeddingModel(),
            embedding_dim=2,
        )
    )

    output = dispatch.fit_transform(_text_table())

    assert output.numerical.size() == (2, 4)
    assert output.columns[Stype.text] == ()


def test_llm_text_embed_deepcopy_shares_embedding_model() -> None:
    embedding_model = torch.nn.Linear(2, 2)
    recipe = Recipe(
        features=ModelTextEmbed(
            embedding_model,
            embedding_dim=2,
        )
    )

    copied = copy.deepcopy(recipe)

    assert copied is not recipe
    assert copied.features is not recipe.features
    assert isinstance(copied.features, ModelTextEmbed)
    assert copied.features.get_submodule("_embedding_model") is embedding_model
