from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from sdm.models.kumo.tabular.model import _KumoTabular

_EXPECTED_ARGS: dict[str, Any] = {
    "task_type": "classification",
    "channels": 128,
    "col_num_heads": 8,
    "row_num_heads": 8,
    "icl_num_heads": 8,
    "feedforward_factor": 2,
    "encoder_layout": "4*(1c 1r)",
    "num_inducing_points": 128,
    "num_cls_tokens": 4,
    "downproject_cls_factor": 1,
    "icl_num_blocks": 12,
    "icl_num_kv_heads_test": None,
    "cell_embedding": "fourier",
    "y_encoder": "one_hot_cls",
    "icl_y_encoder": "one_hot_cls",
    "max_classes": 10,
    "prediction_head": "mlp_cls",
    "norm": "rms",
    "norm_bias": True,
    "activation": "swiglu",
    "ssmax": False,
    "qk_norm": True,
    "per_dim_scale": True,
    "sandwich": True,
    "attention_bias": True,
    "ffn_bias": False,
    "cls_in_column_stage": False,
    "column_stage_affine": True,
    "row_stage_norm": True,
    "rope_interleaved": False,
    "rope_requires_grad": False,
    "rope_frac": 0.25,
    "sdpa_fp32": False,
    "grad_checkpoint": False,
    "text_embedding_dim": 30,
    "icl": "standard",
    "icl_num_thinking_rows": 0,
}


def _insert(
    out: dict[str, Tensor],
    key: str,
    value: Tensor,
    source_key: str,
) -> None:
    if key in out:
        raise ValueError(
            f"Checkpoint entries collide at {key!r}; second source is "
            f"{source_key!r}"
        )
    out[key] = value


def _map_transformer_tail(tail: str, *, has_rope: bool) -> str:
    replacements = {
        "q_norm.": "query_norm.",
        "kv_norm.": "key_value_norm.",
        "mlp_norm.": "mlp.0.",
        "lin1.": "mlp.1.up_lin.",
        "lin_gate.": "mlp.1.gate_lin.",
        "lin2.": "mlp.1.down_lin.",
        "post_mlp_norm.": "mlp.2.",
    }
    for old, new in replacements.items():
        if tail.startswith(old):
            return new + tail.removeprefix(old)

    rope_offset = int(has_rope)
    if tail == "attn.q_norm.weight":
        return f"attn.query_transform.{rope_offset}.weight"
    if tail == "attn.k_norm.weight":
        return f"attn.key_transform.{rope_offset}.weight"
    if tail == "attn.per_dim_scale.scale":
        return f"attn.query_transform.{rope_offset + 1}.weight"
    if tail.startswith(("attn.qkv_lin.", "attn.out_lin.")):
        return tail
    if tail.startswith("post_attn_norm."):
        return tail

    raise KeyError(f"Unsupported transformer checkpoint entry {tail!r}")


def _map_encoder_key(key: str) -> list[str]:
    if key == "encoder.cls_tokens":
        return ["table_encoder.cls_tokens"]
    if key == "encoder.rope.inv_freq":
        return [
            f"table_encoder.row_blocks.{stage}.attn.{side}_transform.0."
            "inv_freq"
            for stage in range(4)
            for side in ("query", "key")
        ]

    prefix = "encoder.stages."
    if not key.startswith(prefix):
        raise KeyError(f"Unsupported encoder checkpoint entry {key!r}")
    stage_text, tail = key.removeprefix(prefix).split(".", 1)
    source_stage = int(stage_text)
    stage = source_stage // 2

    if source_stage % 2 == 0:
        if tail == "blocks.0.inducing_vectors":
            return [f"table_encoder.col_blocks.{stage}.inducing_points"]
        for old, new in (
            ("blocks.0.to_inducing.", "inducing_block."),
            ("blocks.0.from_inducing.", "output_block."),
        ):
            if tail.startswith(old):
                mapped = _map_transformer_tail(
                    tail.removeprefix(old),
                    has_rope=False,
                )
                return [f"table_encoder.col_blocks.{stage}.{new}{mapped}"]
        if tail.startswith("out_lin."):
            return [
                f"table_encoder.col_projections.{stage}.0."
                f"{tail.removeprefix('out_lin.')}"
            ]
        if tail.startswith("out_norm."):
            return [
                f"table_encoder.col_projections.{stage}.1."
                f"{tail.removeprefix('out_norm.')}"
            ]
        raise KeyError(f"Unsupported column-stage checkpoint entry {key!r}")

    if tail.startswith("blocks.0."):
        mapped = _map_transformer_tail(
            tail.removeprefix("blocks.0."),
            has_rope=True,
        )
        return [f"table_encoder.row_blocks.{stage}.{mapped}"]
    if tail.startswith("norm."):
        return [
            f"table_encoder.row_norms.{stage}.{tail.removeprefix('norm.')}"
        ]
    raise KeyError(f"Unsupported row-stage checkpoint entry {key!r}")


def _map_key(key: str) -> list[str]:
    exact = {
        "cell_embedding.frequencies_num": "cell_embedding.num_freq",
        "cell_embedding.frequencies_cat": "cell_embedding.cat_freq",
    }
    if key in exact:
        return [exact[key]]

    for old, new in (
        ("cell_embedding.lin_num.", "cell_embedding.num_lin."),
        ("cell_embedding.lin_cat.", "cell_embedding.cat_lin."),
        ("y_encoder.lin.", "y_encoder."),
        ("icl_y_encoder.lin.", "icl_block.y_lin."),
        ("icl.norm.", "icl_block.norm."),
        ("prediction_head.mlp.", "icl_block.head."),
    ):
        if key.startswith(old):
            return [new + key.removeprefix(old)]

    if key.startswith("encoder."):
        return _map_encoder_key(key)

    if key.startswith("icl.blocks."):
        layer, tail = key.removeprefix("icl.blocks.").split(".", 1)
        return [
            f"icl_block.layers.{layer}."
            f"{_map_transformer_tail(tail, has_rope=False)}"
        ]

    raise KeyError(f"Unsupported checkpoint entry {key!r}")


def _remap_checkpoint(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for source_key, value in state.items():
        if not isinstance(source_key, str) or not isinstance(value, Tensor):
            raise TypeError("Checkpoint state must map string keys to tensors")
        for target_key in _map_key(source_key):
            _insert(out, target_key, value, source_key)
    return out


def _validate_args(args: object) -> None:
    if not isinstance(args, Mapping):
        raise TypeError("Checkpoint 'args' must be a mapping")

    actual = dict(args)
    missing = sorted(key for key in _EXPECTED_ARGS if key not in actual)
    unexpected = sorted(
        repr(key) for key in actual if key not in _EXPECTED_ARGS
    )
    changed = sorted(
        key
        for key, expected in _EXPECTED_ARGS.items()
        if key in actual
        and (
            type(actual[key]) is not type(expected) or actual[key] != expected
        )
    )
    if not missing and not unexpected and not changed:
        return

    raise ValueError(
        "Unsupported Kumo Tabular checkpoint arguments: "
        f"missing={missing}, unexpected={unexpected}, changed={changed}"
    )


def _validate_state(
    state: Mapping[str, Tensor],
    expected: Mapping[str, Tensor],
) -> None:
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            "Checkpoint state keys do not match the model: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for key, value in state.items():
        expected_value = expected[key]
        if value.shape != expected_value.shape:
            raise ValueError(
                f"Checkpoint tensor {key!r} has shape {list(value.shape)}, "
                f"expected {list(expected_value.shape)}"
            )
        if value.dtype != expected_value.dtype:
            raise ValueError(
                f"Checkpoint tensor {key!r} has dtype {value.dtype}, "
                f"expected {expected_value.dtype}"
            )


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str | None = None,
) -> _KumoTabular:
    """Load the supported Kumo Tabular classification checkpoint.

    Args:
        checkpoint_path: Local checkpoint path.
        device: Device of the returned model.

    Returns:
        Checkpoint-loaded classification model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    if "args" not in checkpoint or "model" not in checkpoint:
        raise KeyError("Checkpoint must contain 'args' and 'model'")

    _validate_args(checkpoint["args"])
    raw_state = checkpoint["model"]
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint 'model' must be a mapping")
    state = _remap_checkpoint(cast(Mapping[str, Tensor], raw_state))

    model = _KumoTabular()
    _validate_state(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    return model.to(device=device).eval()
