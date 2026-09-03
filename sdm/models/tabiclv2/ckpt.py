# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

from torch import Tensor


def _map_transformer(prefix: str, tail: str) -> list[str]:
    tail = tail.replace("attn.in_proj_weight", "attn.qkv_lin.weight")
    tail = tail.replace("attn.in_proj_bias", "attn.qkv_lin.bias")
    tail = tail.replace("attn.out_proj.", "attn.out_lin.")
    tail = tail.replace(
        "attn.ssmax_layer.base_mlp.",
        "attn.sdpa.query_scaling.scale.",
    )
    tail = tail.replace(
        "attn.ssmax_layer.query_mlp.",
        "attn.sdpa.query_scaling.gate.",
    )

    if tail.startswith("norm1."):
        tail = tail.removeprefix("norm1.")
        return [
            prefix + "query_norm." + tail,
            prefix + "key_value_norm." + tail,
        ]
    if tail.startswith("norm2."):
        tail = tail.replace("norm2.", "mlp.0.", 1)
    elif tail.startswith("linear1."):
        tail = tail.replace("linear1.", "mlp.1.", 1)
    elif tail.startswith("linear2."):
        tail = tail.replace("linear2.", "mlp.3.", 1)

    return [prefix + tail]


def remap_ckpt(  # noqa: D103
    ckpt: dict[str, Tensor],
    is_classifier: bool,
) -> dict[str, Tensor]:

    out: dict[str, Tensor] = {}

    if is_classifier:
        out["row_embedding.y_emb.weight"] = (
            ckpt["col_embedder.y_encoder.weight"].t()
            + ckpt["col_embedder.y_encoder.bias"]
        )
        out["icl_block.y_emb.weight"] = (
            ckpt["icl_predictor.y_encoder.weight"].t()
            + ckpt["icl_predictor.y_encoder.bias"]
        )
    else:
        out["row_embedding.y_lin.weight"] = ckpt[
            "col_embedder.y_encoder.weight"
        ]
        out["row_embedding.y_lin.bias"] = ckpt["col_embedder.y_encoder.bias"]
        out["icl_block.y_lin.weight"] = ckpt["icl_predictor.y_encoder.weight"]
        out["icl_block.y_lin.bias"] = ckpt["icl_predictor.y_encoder.bias"]

    for key, value in ckpt.items():
        if key.startswith("col_embedder.in_linear."):
            new_key = key.replace(
                "col_embedder.in_linear",
                "row_embedding.lin",
            )
            out[new_key] = value

        elif key.startswith("col_embedder.tf_col.blocks."):
            layer, tail = key.removeprefix(
                "col_embedder.tf_col.blocks."
            ).split(".", 1)

            if tail == "ind_vectors":
                out[f"row_embedding.col_layers.{layer}.inducing_points"] = (
                    value
                )
            elif tail.startswith("multihead_attn1."):
                prefix = f"row_embedding.col_layers.{layer}.inducing_block."
                for new_key in _map_transformer(
                    prefix,
                    tail.removeprefix("multihead_attn1."),
                ):
                    out[new_key] = value
            elif tail.startswith("multihead_attn2."):
                prefix = f"row_embedding.col_layers.{layer}.output_block."
                for new_key in _map_transformer(
                    prefix,
                    tail.removeprefix("multihead_attn2."),
                ):
                    out[new_key] = value

        elif key == "row_interactor.cls_tokens":
            out["row_embedding.readout_token"] = value

        elif key.startswith("row_interactor.tf_row.blocks."):
            layer, tail = key.removeprefix(
                "row_interactor.tf_row.blocks."
            ).split(".", 1)
            prefix = f"row_embedding.row_layers.{layer}."
            for new_key in _map_transformer(prefix, tail):
                out[new_key] = value

        elif key == "row_interactor.tf_row.rope.freqs":
            for layer, side in product(range(3), ("query", "key")):
                out[
                    f"row_embedding.row_layers.{layer}.attn."
                    f"{side}_transform.inv_freq"
                ] = value

        elif key.startswith("row_interactor.out_ln."):
            out[key.replace("row_interactor.out_ln", "row_embedding.norm")] = (
                value
            )

        elif key.startswith("icl_predictor.tf_icl.blocks."):
            layer, tail = key.removeprefix(
                "icl_predictor.tf_icl.blocks."
            ).split(".", 1)
            prefix = f"icl_block.layers.{layer}."
            for new_key in _map_transformer(prefix, tail):
                out[new_key] = value

        elif key.startswith("icl_predictor.ln."):
            out[key.replace("icl_predictor.ln", "icl_block.norm")] = value

        elif key.startswith("icl_predictor.decoder."):
            out[key.replace("icl_predictor.decoder", "icl_block.head")] = value

    return out
