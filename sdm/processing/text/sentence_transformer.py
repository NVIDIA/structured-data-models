# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from typing import cast

import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing.text.llm import LLM


class SentenceTransformer(Processor):
    r"""Embed text columns with a
    :class:`sentence_transformers.SentenceTransformer` model.

    Args:
        model: The model name of a
            :class:`sentence_transformers.SentenceTransformer`.
        postprocess: Optional postprocessing routine, applied independently for
            each text column.
        batch_size: The maximum batch size of the model. Adjusting the batch
            size can significantly improve processing speed. The optimal value
            depends on your hardware, model size, precision, and input length.
        max_batch_tokens: The maximum number of tokens to process per batch.
        max_length: The maximum number of tokens per text before truncation.
        device: The device.
        dtype: The dtype.
    """

    handles_stypes = (Stype.text,)

    def __init__(
        self,
        model: str,
        postprocess: object = None,
        *,
        batch_size: int | None = 32,
        max_batch_tokens: int | None = 65_536,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.requires_fit = False
        if postprocess is not None:
            postprocess = Processor.as_processor(postprocess)
            self.requires_fit = postprocess.requires_fit

        self.model = model
        self.llm = LLM(model, device=device, dtype=dtype)
        self.postprocess = postprocess
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_length = max_length

    def _embed(self, table: TableTensor) -> Tensor:  # [..., R, C, D]
        text = table.text.to_arrow()
        if table.text.is_nullable:
            text = pc.fill_null(text, "")

        D = cast(int, self.llm.model.get_embedding_dimension())
        out = table.numerical.new_empty(
            (len(text), D),
            device=self.llm.model.device,
        )
        for index, emb in self.llm(
            text,
            output_type="sentence",
            batch_size=self.batch_size,
            max_batch_tokens=self.max_batch_tokens,
            max_length=self.max_length,
            deduplicate=True,
            dtype=out.dtype,
        ):
            out[index[1]] = emb[index[0]]

        return out.view(*table.size()[:-1], table.text.size(-1), out.size(-1))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        assert self.postprocess is not None
        self.postprocess.fit(
            TableTensor.from_tensor(self._embed(table).movedim(-2, -3)),
            generator=generator,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        out = self._embed(table)

        if self.postprocess is not None:
            out = self.postprocess.transform(
                TableTensor.from_tensor(out.movedim(-2, -3))
            ).numerical.movedim(-2, -3)  # [..., R, C, D]

        out = TableTensor(
            columns={
                Stype.numerical: tuple(
                    f"{column}__emb{i}"
                    for column in table.text_columns
                    for i in range(out.size(-1))
                )
            },
            numerical=out.flatten(-2).to(table.device),
        )

        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.text), out], dim=-1),
        )

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        out = self._embed(table)

        if self.postprocess is not None:
            out = self.postprocess.fit_transform(
                TableTensor.from_tensor(out.movedim(-2, -3)),
                generator=generator,
            ).numerical.movedim(-2, -3)  # [..., R, C, D]

        out = TableTensor(
            columns={
                Stype.numerical: tuple(
                    f"{column}__emb{i}"
                    for column in table.text_columns
                    for i in range(out.size(-1))
                )
            },
            numerical=out.flatten(-2).to(table.device),
        )

        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.text), out], dim=-1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if self.postprocess is None:
            return (
                f"{' ' * indent}{self.__class__.__name__}("
                f"model={self.model!r})"
            )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{' ' * (indent + 2)}model={self.model!r},\n"
            f"{' ' * (indent + 2)}postprocess="
            f"{self.postprocess.__repr__(indent=indent + 2)[indent + 2 :]},\n"
            f"{' ' * indent})"
        )
