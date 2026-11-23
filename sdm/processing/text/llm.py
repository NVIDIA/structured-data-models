# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, Self, overload

import numpy as np
import pyarrow as pa
import torch
import tqdm
from torch import Tensor

from sdm._warnings import warn_once

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

        if isinstance(model, str):
            model = load_model(
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

    @overload
    def forward(
        self,
        text: pa.Array,
        output_type: Literal["token"],
        batch_size: int | None = ...,
        max_batch_tokens: int | None = ...,
        max_length: int | None = ...,
        deduplicate: bool = ...,
        show_progress: bool = ...,
        device: torch.device | str | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]: ...

    @overload
    def forward(
        self,
        text: pa.Array,
        output_type: Literal["sentence"],
        batch_size: int | None = ...,
        max_batch_tokens: int | None = ...,
        max_length: int | None = ...,
        deduplicate: bool = ...,
        show_progress: bool = ...,
        device: torch.device | str | None = ...,
        dtype: torch.dtype | None = ...,
    ) -> Iterator[tuple[Tensor, Tensor]]: ...

    def forward(
        self,
        text: pa.Array,
        output_type: Literal["token", "sentence"],
        batch_size: int | None = 32,
        max_batch_tokens: int | None = 65_536,
        max_length: int | None = None,
        deduplicate: bool = False,
        show_progress: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Iterator[
        tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor]
    ]:
        if len(text) == 0:
            return

        if max_length is None:
            max_length = self.model.max_seq_length
        elif self.model.max_seq_length is not None:
            max_length = min(max_length, self.model.max_seq_length)

        inverse: Tensor | None = None
        if deduplicate:
            text = text.dictionary_encode()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given NumPy array is not writable",
                )
                inverse = torch.from_numpy(text.indices.to_numpy())
            text = text.dictionary

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
        perm_cpu = length.argsort(descending=True)
        perm = perm_cpu.to(device or self.model.device)
        length = length[perm_cpu].tolist()

        ptr: list[int] | None = None
        edge_index: Tensor | None = None
        if inverse is not None:
            pos = torch.empty_like(perm_cpu)
            pos[perm_cpu] = torch.arange(len(text))
            pos = pos[inverse]
            row = pos.argsort()
            pos = pos[row]

            edge_index = torch.stack([pos, row], dim=0).to(perm.device)
            ptr = torch._convert_indices_from_coo_to_csr(
                pos, len(text)
            ).tolist()

        pbar = tqdm.tqdm(
            total=sum(length),
            desc="Embedding text",
            unit="token",
            unit_scale=True,
            disable=not show_progress,
        )
        start = 0
        while start < len(text):
            count = len(text) - start
            if batch_size is not None:
                count = min(count, batch_size)
            if max_batch_tokens is not None:
                count = min(count, max(1, max_batch_tokens // length[start]))
            end = start + count

            index = perm_cpu[start:end]
            batch = [encodings[i] for i in index]
            width = max(length[start:end])

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
            }
            if output_type == "token":
                inputs["offset_mapping"] = np.asarray(
                    [e.offsets for e in batch],
                    dtype=np.int64,
                )
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

            if edge_index is not None and ptr is not None:
                index = edge_index[:, ptr[start] : ptr[end]]
                index[0] -= start
            else:
                index = perm[start:end]

            with torch.inference_mode():
                out = self.model(inputs)

            if output_type == "token":
                emb = out["token_embeddings"].to(device, dtype)
                offset = inputs["offset_mapping"].to(emb.device)
                attn_mask = inputs["attention_mask"].to(emb.device)
                yield index, emb, offset, attn_mask
            else:
                emb = out["sentence_embedding"][..., : self.model.truncate_dim]
                emb = emb.to(device, dtype)
                yield index, emb

            pbar.update(sum(length[start:end]))
            start = end

        pbar.close()


def load_model(model: str, **kwargs: Any) -> SentenceTransformer:
    import huggingface_hub  # noqa: PLC0415
    import sentence_transformers
    import transformers

    verbosity = transformers.utils.logging.get_verbosity()
    model_enabled = transformers.utils.logging.is_progress_bar_enabled()
    hub_enabled = not huggingface_hub.utils.are_progress_bars_disabled()

    try:
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        huggingface_hub.utils.disable_progress_bars()

        try:
            llm = sentence_transformers.SentenceTransformer(
                model,
                local_files_only=True,
                **kwargs,
            )
            if llm.device.type == "cuda":
                try:
                    llm.transformers_model.use_kernels = True  # type: ignore
                except ValueError as e:
                    warn_once(
                        key="transformers-kernels-missing",
                        message=f"Transformer {e}, for faster LLM execution.",
                    )
            return llm
        except OSError:
            pass
    finally:
        transformers.utils.logging.set_verbosity(verbosity)
        if model_enabled:
            transformers.utils.logging.enable_progress_bar()
        if hub_enabled:
            huggingface_hub.utils.enable_progress_bars()
        else:
            huggingface_hub.utils.disable_progress_bars()

    try:
        transformers.utils.logging.set_verbosity_error()
        llm = sentence_transformers.SentenceTransformer(model, **kwargs)
    finally:
        transformers.utils.logging.set_verbosity(verbosity)
    if llm.device.type == "cuda":
        try:
            llm.transformers_model.use_kernels = True  # type: ignore
        except ValueError as e:
            warn_once(
                key="transformers-kernels-missing",
                message=f"Transformer {e}, for faster LLM execution.",
            )
    return llm
