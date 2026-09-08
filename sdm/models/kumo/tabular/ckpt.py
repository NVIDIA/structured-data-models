"""Remap KumoTFM trainer checkpoints into SDM names."""

from __future__ import annotations

import re
from typing import Any, cast

from torch import Tensor


def unwrap_kumo_checkpoint(
    payload: object,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Return the model state and arguments from a trainer checkpoint."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a checkpoint dict, got {type(payload).__name__}"
        )
    state = payload.get("model", payload)
    args = payload.get("args", {})
    if not isinstance(state, dict) or not isinstance(args, dict):
        raise TypeError("Expected checkpoint model and argument dictionaries")
    return cast(dict[str, Tensor], state), cast(dict[str, Any], args)


def sdm_kwargs_from_kumo_args(args: dict[str, Any]) -> dict[str, Any]:
    """Map KumoTFM arguments to SDM model arguments."""
    channels = int(args.get("channels", 128))
    num_cls = int(args.get("num_cls_tokens", 4))
    downproject = float(args.get("downproject_cls_factor", 1.0))
    layout = str(args.get("encoder_layout", "4*(1c 1r)"))
    kv_heads = args.get("icl_num_kv_heads_test")
    if kv_heads == 0:
        kv_heads = None
    return {
        "cell_channels": channels,
        "num_embedding_layers": _num_encoder_layers(layout),
        "num_embedding_heads": int(
            args.get("col_num_heads") or args.get("row_num_heads") or 4
        ),
        "num_inducing_points": int(args.get("num_inducing_points", 128)),
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": num_cls,
        "icl_channels": max(1, round(num_cls * channels * downproject)),
        "num_icl_layers": int(args.get("icl_num_blocks", 12)),
        "num_icl_heads": int(args.get("icl_num_heads", 8)),
        "num_icl_key_value_heads_for_query": kv_heads,
        "nan_indicator": args.get("cell_embedding") == "fourier_nan_indicator",
        "partial_rotary_factor": float(args.get("rope_frac", 1.0)),
        "row_log_scale": bool(args.get("row_stage_logn_scale")),
    }


def _num_encoder_layers(layout: str) -> int:
    layout = " ".join(layout.split())
    match = re.fullmatch(r"(\d+)\*\(1c 1r\)", layout)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"(\d+)c (\d+)r", layout)
    if match:
        return int(match.group(1))
    return 4


def _map_transformer_tail(tail: str) -> str | None:
    """Map one kumo-scm TransformerBlock suffix onto the SDM block."""
    replacements = (
        ("q_norm.", "query_norm."),
        ("kv_norm.", "key_value_norm."),
        ("attn.sdpa.per_head_logn_scale.", "attn.sdpa.query_scaling."),
        ("attn.sdpa.gated_per_head_logn_scale.", "attn.sdpa.query_scaling."),
        ("mlp_norm.", "mlp.0."),
        ("lin1.", "mlp.1."),
        ("lin2.", "mlp.3."),
    )
    for old, new in replacements:
        if tail.startswith(old):
            return tail.replace(old, new, 1)
    if tail.startswith(("attn.qkv_lin.", "attn.out_lin.")):
        return tail
    # Head-wise q/k RMSNorm is affine-free in this architecture.
    if tail.startswith(("attn.q_norm.", "attn.k_norm.", "attn.sdpa.ssmax")):
        return None
    return tail


def remap_kumo_tfm_ckpt(
    ckpt: dict[str, Tensor],
    *,
    is_classifier: bool,
    num_row_layers: int,
) -> dict[str, Tensor]:
    """Map a kumo-scm ``KumoTFM`` state dict onto :class:`_KumoTabular`."""
    out: dict[str, Tensor] = {}
    rope: Tensor | None = ckpt.get("encoder.rope.inv_freq")

    for key, value in ckpt.items():
        if key == "cell_embedding.frequencies_num":
            out["row_embedding.cell_embedding.num_freq"] = value
            continue
        if key == "cell_embedding.frequencies_cat":
            out["row_embedding.cell_embedding.cat_freq"] = value
            continue
        if key.startswith("cell_embedding.lin_num."):
            key = key.replace("lin_num.", "num_lin.", 1)
            out[f"row_embedding.{key}"] = value
            continue
        if key.startswith("cell_embedding.lin_cat."):
            key = key.replace("lin_cat.", "cat_lin.", 1)
            out[f"row_embedding.{key}"] = value
            continue
        if key.startswith("cell_embedding.lin_missing."):
            out[f"row_embedding.{key}"] = value
            continue

        if key.startswith("y_encoder."):
            tail = key.removeprefix("y_encoder.")
            if is_classifier:
                if tail in {"weight", "lin.weight"}:
                    bias = ckpt.get(
                        "y_encoder.bias",
                        ckpt.get("y_encoder.lin.bias"),
                    )
                    out["row_embedding.y_emb.weight"] = (
                        value.T if bias is None else value.T + bias
                    )
                continue
            if tail.startswith("lin."):
                out[f"row_embedding.y_lin.{tail.removeprefix('lin.')}"] = value
            continue

        if key.startswith("icl_y_encoder."):
            tail = key.removeprefix("icl_y_encoder.")
            if is_classifier:
                if tail in {"weight", "lin.weight"}:
                    bias = ckpt.get(
                        "icl_y_encoder.bias",
                        ckpt.get("icl_y_encoder.lin.bias"),
                    )
                    out["icl_block.y_emb.weight"] = (
                        value.T if bias is None else value.T + bias
                    )
                continue
            if tail.startswith("lin."):
                out[f"icl_block.y_lin.{tail.removeprefix('lin.')}"] = value
            continue

        if key == "encoder.cls_tokens":
            out["row_embedding.readout_token"] = value
            continue
        if key == "encoder.rope.inv_freq":
            continue
        if key.startswith("encoder.norm."):
            out[key.replace("encoder.norm.", "row_embedding.norm.", 1)] = value
            continue

        if key.startswith("encoder.stages."):
            rest = key.removeprefix("encoder.stages.")
            stage_s, rest = rest.split(".", 1)
            stage = int(stage_s)
            layer = stage // 2
            is_column = stage % 2 == 0
            if rest.startswith("blocks.0."):
                rest = rest.removeprefix("blocks.0.")
            if is_column:
                if rest == "inducing_vectors":
                    out[
                        f"row_embedding.col_blocks.{layer}.inducing_points"
                    ] = value
                    continue
                if rest.startswith("to_inducing."):
                    mapped = _map_transformer_tail(
                        rest.removeprefix("to_inducing.")
                    )
                    if mapped is not None:
                        out[
                            f"row_embedding.col_blocks.{layer}.inducing_block."
                            f"{mapped}"
                        ] = value
                    continue
                if rest.startswith("from_inducing."):
                    mapped = _map_transformer_tail(
                        rest.removeprefix("from_inducing.")
                    )
                    if mapped is not None:
                        out[
                            f"row_embedding.col_blocks.{layer}.output_block."
                            f"{mapped}"
                        ] = value
                    continue
            else:
                mapped = _map_transformer_tail(rest)
                if mapped is not None:
                    out[f"row_embedding.row_blocks.{layer}.{mapped}"] = value
                continue

        if key.startswith("cls_downproject."):
            out[key.replace("cls_downproject.", "row_project.", 1)] = value
            continue

        if key.startswith(("icl.blocks.", "icl.test_blocks.")):
            prefix = (
                "icl.blocks."
                if key.startswith("icl.blocks.")
                else "icl.test_blocks."
            )
            rest = key.removeprefix(prefix)
            layer_s, tail = rest.split(".", 1)
            mapped = _map_transformer_tail(tail)
            if mapped is not None:
                out[f"icl_block.layers.{layer_s}.{mapped}"] = value
            continue
        if key.startswith("icl.train_blocks."):
            continue
        if key.startswith("icl.norm."):
            out[key.replace("icl.norm.", "icl_block.norm.", 1)] = value
            continue

        if key.startswith("prediction_head.mlp."):
            out[key.replace("prediction_head.mlp.", "icl_block.head.", 1)] = (
                value
            )
            continue
        if key.startswith("prediction_head."):
            continue

    if rope is not None:
        for layer in range(num_row_layers):
            for side in ("query", "key"):
                out[
                    f"row_embedding.row_blocks.{layer}.attn.{side}_transform.0."
                    f"inv_freq"
                ] = rope

    return out
