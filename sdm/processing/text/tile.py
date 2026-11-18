# Copyright (c) 2025, Félix Lefebvre
# Licensed under the BSD 3-Clause License; see https://github.com/soda-inria/tile/blob/main/LICENSE

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

import math
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing.text.llm import LLM

MODELS: dict[str, tuple[str, int]] = {  # (model_name, num_layers)
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B", 14),
    "llama-3.1-70b": ("meta-llama/Llama-3.1-70B", 32),
    "llama-3.2-1b": ("meta-llama/Llama-3.2-1B", 10),
    "llama-3.2-3b": ("meta-llama/Llama-3.2-3B", 11),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B-Base", 16),
    "qwen3-4b": ("Qwen/Qwen3-4B-Base", 20),
    "qwen3-8b": ("Qwen/Qwen3-8B-Base", 19),
}


def textify(table: TableTensor) -> tuple[pa.Array, Tensor]:
    """Join cells as ``The {column} is {value}.`` sentences per row.

    Args:
        table: The input table with shape ``[..., C]``, where ``C`` is the
            number of columns.

    Returns:
        A flat :class:`pyarrow.Array` of row strings and a CPU tensor of
        character spans with shape ``[..., C, 2]``.
    """
    dtype = pa.large_string()
    if table.size(-1) == 0:
        array = pa.repeat(pa.scalar("", dtype), math.prod(table.size()[:-1]))
        span = torch.empty((*table.size(), 2), dtype=torch.int64)
        return (array, span)

    _table = table.to_arrow()
    cells, counts = [], []
    for column, array in zip(_table.column_names, _table.itercolumns()):
        if pa.types.is_floating(array.type):
            array = pc.if_else(pc.is_nan(array), None, array)  # type: ignore

        array = array.cast(dtype)
        array = pc.fill_null(array, pa.scalar("missing", dtype))
        cell = pc.binary_join_element_wise(  # type: ignore
            pa.scalar(f"The {column} is ", dtype),
            array,
            pa.scalar(".", dtype),
            pa.scalar("", dtype),
        )
        cells.append(cell)
        counts.append(pc.utf8_length(cell).to_numpy())  # type: ignore

    count = torch.from_numpy(np.stack(counts, axis=-1)).view(table.size())
    end = (count + 1).cumsum(dim=-1) - 1
    span = torch.stack([end - count, end], dim=-1)
    array = pc.binary_join_element_wise(  # type: ignore
        *cells,
        pa.scalar(" ", dtype),
    )
    return array.combine_chunks(), span


class TILE(Processor):
    r"""The text cell embedding module from the `"TILE: Bridging LLMs and Table
    Foundation Models for Tabular Learning with Strings"
    <https://arxiv.org/abs/2610.17065>`_ paper.

    Each row is serialized as one sentence per cell. An LLM embeds the row,
    and token embeddings are mean-pooled for each text cell. Optional
    postprocessing transforms these embeddings independently per text
    column, for example to reduce dimensionality.

    Args:
        model: The LLM, one of ``"llama-3.1-8b"``, ``"llama-3.1-70b"``,
            ``"llama-3.2-1b"``, ``"llama-3.2-3b"``, ``"qwen3-0.6b"``,
            ``"qwen3-4b"``, or ``"qwen3-8b"``.
        postprocess: Optional postprocessing routine, applied independently for
            each text column.
        batch_size: The batch size of the LLM. Adjusting the batch size can
            significantly improve processing speed. The optimal value depends
            on your hardware, model size, precision, and input length.
        max_length: Maximum number of tokens per row.
        device: The LLM device.
        dtype: The LLM dtype.
    """

    handles_stypes = (Stype.text,)

    def __init__(
        self,
        model: str = "qwen3-0.6b",
        postprocess: object = None,
        *,
        batch_size: int = 64,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        if model not in MODELS:
            raise ValueError(
                f"Unsupported model {model!r}. Supported models: "
                f"{', '.join(MODELS)}"
            )

        from sentence_transformers import SentenceTransformer
        from transformers import AutoConfig

        model, num_layers = MODELS[model]
        config_kwargs: dict[str, object] = {
            "num_hidden_layers": num_layers,
        }
        config = AutoConfig.from_pretrained(model)
        if getattr(config, "layer_types", None) is not None:
            config_kwargs["layer_types"] = config.layer_types[:num_layers]

        llm = SentenceTransformer(
            model,
            device=str(device) if device is not None else None,
            config_kwargs=config_kwargs,
            model_kwargs={"dtype": dtype},
        )
        cast(torch.nn.Module, llm.transformers_model).set_submodule(
            target="norm",
            module=torch.nn.Identity(),
        )

        self.requires_fit = False
        if postprocess is not None:
            postprocess = Processor.as_processor(postprocess)
            self.requires_fit = postprocess.requires_fit

        self.llm = LLM(llm)
        self.postprocess = postprocess
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_first = len(self.llm.model.tokenizer("")["input_ids"]) == 0

    def _embed(self, table: TableTensor) -> Tensor:  # [..., R, C, D]
        columns = tuple(
            column for columns in table.columns.values() for column in columns
        )
        positions = [columns.index(column) for column in table.text_columns]

        text, span = textify(table)
        span = span.view(len(text), table.size(-1), 2)[:, positions]
        span = span.to(self.llm.model.device)

        D = self.llm.model.transformers_model.config.hidden_size  # type: ignore
        out = table.numerical.new_empty(
            (len(text), len(positions), D),
            device=self.llm.model.device,
        )
        for (  # Mean pool embeddings for each text cell:
            index,  # [R]
            emb,  # [R, T, D]
            offset,  # [R, T, 2]
            mask,  # [R, T]
        ) in self.llm(text, self.batch_size, self.max_length):
            emb = emb.to(out.dtype)
            start, end = span[index].unbind(dim=-1)
            mask = mask & (offset[..., 1] > offset[..., 0])
            if self.skip_first:
                mask[..., 0] = False
            weight = (  # Match tokens to text cells:
                mask[:, None, :]
                & (offset[:, None, :, 0] < end[:, :, None])
                & (offset[:, None, :, 1] > start[:, :, None])
            ).to(out.dtype)  # [R, C, T]
            weight /= weight.sum(dim=-1, keepdim=True).clamp(min=1)
            out[index] = weight.bmm(emb)  # [R, C, D]

        return out.view(*table.size()[:-1], *out.size()[-2:])  # [..., R, C, D]

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if self.postprocess is None:
            return

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
