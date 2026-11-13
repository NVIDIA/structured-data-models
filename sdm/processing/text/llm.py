# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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

    def forward(
        self,
        text: pa.Array,
        batch_size: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:

        if len(text) > batch_size:
            perm = np.argsort(-pc.utf8_length(text).to_numpy())  # type: ignore
            text = text.take(pa.array(perm))
            perm = torch.from_numpy(perm).to(device or self.model.device)
        else:
            perm = torch.arange(len(text), device=device or self.model.device)

        for start in range(0, len(text), batch_size):
            inputs = self.model.tokenizer(
                text=text.slice(start, batch_size).to_pylist(),
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
                return_offsets_mapping=True,
                return_attention_mask=True,
                return_tensors=None,
            )
            for key, value in inputs.items():
                inputs[key] = torch.from_numpy(np.asarray(value))

            inputs = inputs.to(self.model.device)
            inputs["use_cache"] = False

            with torch.inference_mode():
                emb = self.model(inputs)["token_embeddings"]

            emb = emb.to(device, dtype)
            offset = inputs["offset_mapping"].to(emb.device)
            attn_mask = inputs["attention_mask"].to(emb.device, torch.bool)

            yield perm[start : start + batch_size], emb, offset, attn_mask
