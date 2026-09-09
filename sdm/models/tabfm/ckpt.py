# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from collections import defaultdict
from itertools import product

import torch
from torch import Tensor


def _map_transformer(prefix: str, tail: str, *, has_rope: bool) -> list[str]:
    if tail.startswith("pre_attn_ln."):
        tail = tail.removeprefix("pre_attn_ln.")
        return [
            f"{prefix}query_norm.{tail}",
            f"{prefix}key_value_norm.{tail}",
        ]
    if tail.startswith("post_attn_ln."):
        tail = tail.replace("post_attn_ln.", "post_attn_norm.", 1)
    elif tail.startswith("pre_ff_ln."):
        tail = tail.replace("pre_ff_ln.", "mlp.0.", 1)
    elif tail.startswith("linear1."):
        tail = tail.replace("linear1.", "mlp.1.up_lin.", 1)
    elif tail.startswith("linear1_gate."):
        tail = tail.replace("linear1_gate.", "mlp.1.gate_lin.", 1)
    elif tail.startswith("linear2."):
        tail = tail.replace("linear2.", "mlp.1.down_lin.", 1)
    elif tail.startswith("post_ff_ln."):
        tail = tail.replace("post_ff_ln.", "mlp.2.", 1)
    elif tail.startswith("attn.out_proj."):
        tail = tail.replace("attn.out_proj.", "attn.out_lin.", 1)
    elif tail == "attn.query_ln.weight":
        index = int(has_rope)
        tail = f"attn.query_transform.{index}.weight"
    elif tail == "attn.key_ln.weight":
        index = int(has_rope)
        tail = f"attn.key_transform.{index}.weight"
    elif tail == "attn.per_dim_scale":
        index = int(has_rope) + 1
        tail = f"attn.query_transform.{index}.weight"

    return [prefix + tail]


def _add_transformer(
    out: dict[str, Tensor],
    qkv: dict[tuple[str, str], dict[str, Tensor]],
    prefix: str,
    tail: str,
    value: Tensor,
    *,
    has_rope: bool,
) -> None:
    match = re.match(r"attn\.([qkv])_proj\.(weight|bias)", tail)
    if match is not None:
        projection, parameter = match.groups()
        qkv[(prefix, parameter)][projection] = value
        return

    for key in _map_transformer(prefix, tail, has_rope=has_rope):
        out[key] = value


def _add_col_embedding(
    out: dict[str, Tensor],
    qkv: dict[tuple[str, str], dict[str, Tensor]],
    repeat: int,
    tail: str,
    value: Tensor,
) -> None:
    if tail.startswith("tf_col.blocks."):
        layer, tail = tail.removeprefix("tf_col.blocks.").split(".", 1)
        if tail == "ind_vectors":
            out[
                f"row_embedding.col_blocks.{repeat}.{layer}.inducing_points"
            ] = value
            return

        block, tail = tail.split(".", 1)
        block = {"mab1": "inducing_block", "mab2": "output_block"}[block]
        _add_transformer(
            out,
            qkv,
            f"row_embedding.col_blocks.{repeat}.{layer}.{block}.",
            tail,
            value,
            has_rope=False,
        )
        return

    if tail.startswith("out_w."):
        out[
            f"row_embedding.col_projections.{repeat}.0."
            f"{tail.removeprefix('out_w.')}"
        ] = value
    elif tail.startswith("ln_w."):
        out[
            f"row_embedding.col_projections.{repeat}.1."
            f"{tail.removeprefix('ln_w.')}"
        ] = value


def _add_row_embedding(
    out: dict[str, Tensor],
    qkv: dict[tuple[str, str], dict[str, Tensor]],
    repeat: int,
    tail: str,
    value: Tensor,
) -> None:
    if tail.startswith("tf_row.blocks."):
        layer, tail = tail.removeprefix("tf_row.blocks.").split(".", 1)
        _add_transformer(
            out,
            qkv,
            f"row_embedding.row_blocks.{repeat}.{layer}.",
            tail,
            value,
            has_rope=True,
        )
    elif tail.startswith("out_ln."):
        out[
            f"row_embedding.row_norms.{repeat}.{tail.removeprefix('out_ln.')}"
        ] = value


def remap_ckpt(  # noqa: D103
    ckpt: dict[str, Tensor],
    is_classifier: bool,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qkv: dict[tuple[str, str], dict[str, Tensor]] = defaultdict(dict)

    if is_classifier:
        class_weight = ckpt["icl_predictor.y_encoder.projection.weight"]
        class_bias = ckpt["icl_predictor.y_encoder.projection.bias"]
        out["icl_block.y_emb.weight"] = class_weight.t() + class_bias

    for repeat, layer, side in product(range(2), range(3), ("query", "key")):
        out[
            f"row_embedding.row_blocks.{repeat}.{layer}.attn."
            f"{side}_transform.0.inv_freq"
        ] = ckpt["row_interactor.tf_row.rope.freqs"]

    for key, value in ckpt.items():
        if key == "cell_embedder.fourier_frequencies":
            out["row_embedding.cell_embedding.num_freq"] = value

        elif key == "cell_embedder.fourier_frequencies_cat":
            out["row_embedding.cell_embedding.cat_freq"] = value

        elif key.startswith("cell_embedder.in_linear."):
            out[
                key.replace(
                    "cell_embedder.in_linear",
                    "row_embedding.cell_embedding.num_lin",
                )
            ] = value

        elif key.startswith("cell_embedder.in_linear_cat."):
            out[
                key.replace(
                    "cell_embedder.in_linear_cat",
                    "row_embedding.cell_embedding.cat_lin",
                )
            ] = value

        elif key == "cell_embedder.y_embedder_lookup.weight":
            if is_classifier:
                out["row_embedding.y_emb.weight"] = value

        elif key.startswith("cell_embedder.y_embedder_lookup.layers."):
            if not is_classifier:
                layer, tail = key.removeprefix(
                    "cell_embedder.y_embedder_lookup.layers."
                ).split(".", 1)
                out[f"row_embedding.y_mlp.{2 * int(layer)}.{tail}"] = value

        elif key == "cls_tokens":
            out["row_embedding.readout_token"] = value

        elif key.startswith(("col_embedder.", "col_embedder_2.")):
            root, tail = key.split(".", 1)
            repeat = int(root == "col_embedder_2")
            _add_col_embedding(out, qkv, repeat, tail, value)

        elif key.startswith(("row_interactor.", "row_interactor_2.")):
            root, tail = key.split(".", 1)
            repeat = int(root == "row_interactor_2")
            _add_row_embedding(out, qkv, repeat, tail, value)

        elif key.startswith("icl_predictor.y_encoder.layers."):
            if not is_classifier:
                layer, tail = key.removeprefix(
                    "icl_predictor.y_encoder.layers."
                ).split(".", 1)
                out[f"icl_block.y_mlp.{2 * int(layer)}.{tail}"] = value

        elif key.startswith("icl_predictor.tf_icl.blocks."):
            layer, tail = key.removeprefix(
                "icl_predictor.tf_icl.blocks."
            ).split(".", 1)
            _add_transformer(
                out,
                qkv,
                f"icl_block.layers.{layer}.",
                tail,
                value,
                has_rope=False,
            )

        elif key.startswith("icl_predictor.ln."):
            out[key.replace("icl_predictor.ln", "icl_block.norm", 1)] = value

        elif key.startswith("icl_predictor.decoder.layers."):
            layer, tail = key.removeprefix(
                "icl_predictor.decoder.layers."
            ).split(".", 1)
            out[f"icl_block.head.{2 * int(layer)}.{tail}"] = value

    for (prefix, parameter), parts in qkv.items():
        if all(projection in parts for projection in ("q", "k", "v")):
            out[f"{prefix}attn.qkv_lin.{parameter}"] = torch.cat(
                [parts[projection] for projection in ("q", "k", "v")],
                dim=0,
            )

    return out
