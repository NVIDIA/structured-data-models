from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import StypeDispatch
from sdm.processing.text.llm_text_embed import LLMTextEmbed
from sdm.testing import onlyCUDA

if TYPE_CHECKING:
    import cudf


class _FakeEmbedder:
    """Deterministic embedder: [len(s), len(s) ** 2] per string."""

    dim = 2

    def encode(self, strings: pa.Array) -> Tensor:
        lengths = pc.call_function("utf8_length", [strings])
        values = torch.tensor(lengths.to_numpy(zero_copy_only=False))
        values = values.to(torch.get_default_dtype())
        return torch.stack((values, values.square()), dim=-1)


class _WrongShapeEmbedder:
    dim = 2

    def encode(self, strings: pa.Array) -> Tensor:
        return torch.zeros(len(strings) + 1, 2)


class _CudfEmbedder:
    dim = 2

    def encode(self, strings: cudf.Series) -> Tensor:
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
    output = LLMTextEmbed(_FakeEmbedder()).transform(_text_table())

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


def test_llm_text_embed_rejects_wrong_encode_shape() -> None:
    with pytest.raises(ValueError, match="Expected 'encode'"):
        LLMTextEmbed(_WrongShapeEmbedder()).transform(_text_table())


def test_llm_text_embed_empty_rows_use_dim_without_encode() -> None:
    table = TableTensor(
        columns={"text": ("title",)},
        text=StringTensor(
            data=torch.empty(0, dtype=torch.uint8),
            offset=torch.zeros(1, dtype=torch.int32),
            size=(0, 1),
        ),
    )

    output = LLMTextEmbed(_FakeEmbedder()).transform(table)

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

    output = LLMTextEmbed(_CudfEmbedder()).transform(table)

    assert output.numerical.device.type == "cuda"
    assert output.columns[Stype.numerical] == ("title_0", "title_1")
    torch.testing.assert_close(
        output.numerical,
        torch.tensor([[2.0, 4.0], [4.0, 16.0]], device="cuda"),
    )


def test_llm_text_embed_in_stype_dispatch_route() -> None:
    dispatch = StypeDispatch(text=LLMTextEmbed(_FakeEmbedder()))

    output = dispatch.fit_transform(_text_table())

    assert output.numerical.size() == (2, 4)
    assert output.columns[Stype.text] == ()
