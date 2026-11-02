# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import cast

import torch
from torch import Tensor

import sdm
import sdm.processing as sp


@torch.inference_mode()
def embed_text(
    table: sdm.TableTensor,
    text_model: str,
    text_dim: int,
    mask: Tensor | None = None,
    device: torch.device | str | None = None,
) -> sdm.TableTensor:
    r"""Embed all text columns of all masked rows."""
    if len(table.text_columns) == 0:
        return table

    text = table.select_stypes("text")
    if mask is not None:
        text = text[mask]
    text = cast(sdm.TableTensor, text.to(device))

    processor = sp.Sequential(
        sp.SentenceTransformer(
            text_model,
            batch_size=32,
            max_seq_length=2048,
        ),
        sp.PCA(text_dim),
    )

    print(f"Embedding {text.size(-1):,} text columns... ", end="", flush=True)
    t = time.perf_counter()
    out = processor.fit_transform(text).to(table.device)
    print(f"Done! [{time.perf_counter() - t:.2f}s]")

    if mask is not None:
        emb = out.numerical.new_full(
            (table.size(0), out.size(-1)),
            fill_value=torch.nan,
        )
        emb[mask] = out.numerical
        out = sdm.TableTensor(columns=out.columns, numerical=emb)

    return cast(
        sdm.TableTensor,
        torch.cat([table.drop_stypes("text"), out], dim=-1),
    )
