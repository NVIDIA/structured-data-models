# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class LLM(torch.nn.Module):
    def __init__(
        self,
        model: str | SentenceTransformer,
        *,
        padding_side: Literal["auto", "left", "right"] = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        from sentence_transformers import SentenceTransformer

        if isinstance(model, str):
            model = SentenceTransformer(
                model,
                device=str(device) if device is not None else None,
                model_kwargs={"dtype": dtype} if dtype is not None else None,
            )
        if padding_side != "auto":
            model.tokenizer.padding_side = padding_side
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        self.model = model.to(device, dtype).eval()

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        return self

    def forward(
        self,
        text: pa.Array,
        batch_size: int = 64,
        max_length: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:

        encodings = self.model.tokenizer(
            text=text.to_pylist(),
            padding=False,
            truncation=max_length is not None,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors=None,
        ).encodings

        length = np.fromiter(map(len, encodings), dtype=np.int64)
        length = torch.from_numpy(length)
        if len(text) > batch_size:
            perm_cpu = length.argsort()
        else:
            perm_cpu = torch.arange(length.numel())
        perm = perm_cpu.to(device or self.model.device)

        for start in range(0, len(text), batch_size):
            index = perm_cpu[start : start + batch_size]
            batch = [encodings[i] for i in index]
            width = int(length[index].max())

            for encoding in batch:
                encoding.pad(
                    length=width,
                    direction=self.model.tokenizer.padding_side,
                    pad_id=self.model.tokenizer.pad_token_id,
                    pad_type_id=self.model.tokenizer.pad_token_type_id,
                    pad_token=self.model.tokenizer.pad_token,
                )

            inputs = {
                "input_ids": np.asarray(
                    [e.ids for e in batch],
                    dtype=np.int64,
                ),
                "attention_mask": np.asarray(
                    [e.attention_mask for e in batch],
                    dtype=np.bool,
                ),
                "offset_mapping": np.asarray(
                    [e.offsets for e in batch],
                    dtype=np.int64,
                ),
            }
            if "token_type_ids" in self.model.tokenizer.model_input_names:
                inputs["token_type_ids"] = np.asarray(
                    [e.type_ids for e in batch],
                    dtype=np.int64,
                )

            inputs = {
                key: torch.from_numpy(value).to(self.model.device)
                for key, value in inputs.items()
            }
            inputs["use_cache"] = False  # type: ignore

            with torch.inference_mode():
                emb = self.model(inputs)["token_embeddings"]

            emb = emb.to(device, dtype)
            offset = inputs["offset_mapping"].to(emb.device)
            attn_mask = inputs["attention_mask"].to(emb.device)

            yield perm[start : start + batch_size], emb, offset, attn_mask
