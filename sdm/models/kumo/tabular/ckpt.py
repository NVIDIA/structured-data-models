# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor


def remap_ckpt(  # noqa: D103
    ckpt: dict[str, Tensor],
    is_classifier: bool,
) -> dict[str, Tensor]:

    out: dict[str, Tensor] = {}
    for key, value in ckpt.items():
        if key == "y_encoder.weight":
            if is_classifier:
                out["row_embedding.y_emb.weight"] = (
                    value.T + ckpt["y_encoder.bias"]
                )
            else:
                out["row_embedding.y_lin.weight"] = value
            continue

        if is_classifier and key == "y_encoder.bias":
            continue

        if key == "table_encoder.cls_tokens":
            out["row_embedding.readout_token"] = value
            continue

        if key.startswith("table_encoder."):
            key = key.removeprefix("table_encoder.")
            out[f"row_embedding.{key}"] = value
            continue

        if key.startswith("cls_downproject."):
            key = key.removeprefix("cls_downproject.")
            out[f"row_project.{key}"] = value
            continue

        if key.startswith("cell_embedding."):
            out[f"row_embedding.{key}"] = value
            continue

        out[key] = value

    return out
