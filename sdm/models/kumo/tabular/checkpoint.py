from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
from torch import Tensor

Task = Literal["classification", "regression"]
_Model = TypeVar("_Model", bound=torch.nn.Module)

_BASE_ARGS: dict[str, Any] = {
    "channels": 128,
    "col_num_heads": 4,
    "row_num_heads": 4,
    "icl_num_heads": 8,
    "feedforward_factor": 2,
    "encoder_layout": "4*(1c 1r)",
    "num_inducing_points": 128,
    "num_cls_tokens": 4,
    "downproject_cls_factor": 1.0,
    "icl_num_blocks": 12,
    "icl_num_kv_heads_test": None,
    "cell_embedding": "fourier_nan_indicator",
    "max_classes": 10,
    "norm": "rms",
    "norm_bias": True,
    "activation": "gelu",
    "ssmax": False,
    "qk_norm": True,
    "qk_norm_affine": False,
    "per_dim_scale": False,
    "per_dim_scale_per_head": False,
    "gated_logn_scale": False,
    "per_head_logn_scale": True,
    "row_stage_logn_scale": True,
    "sandwich": False,
    "attention_bias": True,
    "ffn_bias": True,
    "cls_in_column_stage": False,
    "column_stage_affine": False,
    "row_stage_norm": False,
    "rope_interleaved": False,
    "rope_requires_grad": False,
    "rope_frac": 1.0,
    "sdpa_fp32": False,
    "grad_checkpoint": False,
    "text_embedding_dim": 30,
    "icl": "standard",
    "icl_num_thinking_rows": 0,
}
_EXPECTED_ARGS_BY_TASK: dict[Task, dict[str, Any]] = {
    "classification": {
        **_BASE_ARGS,
        "task_type": "classification",
        "y_encoder": "one_hot_cls",
        "icl_y_encoder": "one_hot_cls",
        "prediction_head": "mlp_cls",
    },
    "regression": {
        **_BASE_ARGS,
        "task_type": "regression",
        "y_encoder": "linear_reg",
        "icl_y_encoder": "linear_reg",
        "prediction_head": "quantile_reg",
    },
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


def _map_transformer_tail(tail: str) -> str:
    replacements = {
        "q_norm.": "query_norm.",
        "kv_norm.": "key_value_norm.",
        "mlp_norm.": "mlp.0.",
        "lin1.": "mlp.1.",
        "lin2.": "mlp.3.",
        "attn.sdpa.per_head_logn_scale.": "attn.sdpa.query_scaling.",
        "attn.sdpa.gated_per_head_logn_scale.": ("attn.sdpa.query_scaling."),
    }
    for old, new in replacements.items():
        if tail.startswith(old):
            return new + tail.removeprefix(old)

    if tail.startswith(("attn.qkv_lin.", "attn.out_lin.")):
        return tail
    raise KeyError(f"Unsupported transformer checkpoint entry {tail!r}")


def _map_encoder_key(key: str) -> list[str]:
    if key == "encoder.cls_tokens":
        return ["row_embedding.readout_token"]
    if key == "encoder.rope.inv_freq":
        return [
            f"row_embedding.row_blocks.{stage}.attn.{side}_transform.0."
            "inv_freq"
            for stage in range(4)
            for side in ("query", "key")
        ]
    if key == "encoder.norm.weight":
        return ["row_embedding.norm.weight"]

    prefix = "encoder.stages."
    if not key.startswith(prefix):
        raise KeyError(f"Unsupported encoder checkpoint entry {key!r}")
    stage_text, tail = key.removeprefix(prefix).split(".", 1)
    source_stage = int(stage_text)
    if source_stage not in range(8):
        raise KeyError(
            f"Unsupported encoder stage in checkpoint entry {key!r}"
        )
    stage = source_stage // 2

    if source_stage % 2 == 0:
        if tail == "blocks.0.inducing_vectors":
            return [f"row_embedding.col_blocks.{stage}.inducing_points"]
        for old, new in (
            ("blocks.0.to_inducing.", "inducing_block."),
            ("blocks.0.from_inducing.", "output_block."),
        ):
            if tail.startswith(old):
                mapped = _map_transformer_tail(tail.removeprefix(old))
                return [f"row_embedding.col_blocks.{stage}.{new}{mapped}"]
        raise KeyError(f"Unsupported column-stage checkpoint entry {key!r}")

    prefix = "blocks.0."
    if tail.startswith(prefix):
        mapped = _map_transformer_tail(tail.removeprefix(prefix))
        return [f"row_embedding.row_blocks.{stage}.{mapped}"]
    raise KeyError(f"Unsupported row-stage checkpoint entry {key!r}")


def _map_key(key: str) -> list[str]:
    exact = {
        "cell_embedding.frequencies_num": (
            "row_embedding.cell_embedding.num_freq"
        ),
        "cell_embedding.frequencies_cat": (
            "row_embedding.cell_embedding.cat_freq"
        ),
        "cell_embedding.lin_missing.weight": (
            "row_embedding.cell_embedding.nan_lin.weight"
        ),
        "icl.norm.weight": "icl_block.norm.weight",
    }
    if key in exact:
        return [exact[key]]

    for old, new in (
        (
            "cell_embedding.lin_num.",
            "row_embedding.cell_embedding.num_lin.",
        ),
        (
            "cell_embedding.lin_cat.",
            "row_embedding.cell_embedding.cat_lin.",
        ),
        ("prediction_head.mlp.", "icl_block.head."),
    ):
        if key.startswith(old):
            return [new + key.removeprefix(old)]

    if key.startswith("encoder."):
        return _map_encoder_key(key)

    if key.startswith("icl.blocks."):
        layer, tail = key.removeprefix("icl.blocks.").split(".", 1)
        return [f"icl_block.layers.{layer}.{_map_transformer_tail(tail)}"]

    raise KeyError(f"Unsupported checkpoint entry {key!r}")


def _validate_alphas(alphas: Tensor) -> None:
    expected = torch.linspace(
        0.001,
        0.999,
        999,
        device=alphas.device,
        dtype=torch.float32,
    )
    if alphas.shape != expected.shape:
        raise ValueError(
            "Checkpoint tensor 'prediction_head.alphas' has shape "
            f"{list(alphas.shape)}, expected {list(expected.shape)}"
        )
    if alphas.dtype != expected.dtype:
        raise ValueError(
            "Checkpoint tensor 'prediction_head.alphas' has dtype "
            f"{alphas.dtype}, expected {expected.dtype}"
        )
    if not torch.equal(alphas, expected):
        raise ValueError("Checkpoint quantile alphas do not match the model")


def _classification_embedding(
    state: Mapping[str, Tensor],
    prefix: str,
) -> Tensor:
    weight_key = f"{prefix}.lin.weight"
    bias_key = f"{prefix}.lin.bias"
    weight = state.get(weight_key)
    bias = state.get(bias_key)
    if not isinstance(weight, Tensor) or not isinstance(bias, Tensor):
        raise ValueError(
            f"Classification checkpoint requires {weight_key!r} and "
            f"{bias_key!r}"
        )
    return weight.T + bias


def _remap_checkpoint(
    state: Mapping[str, Tensor],
    *,
    task: Task,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    has_alphas = False
    for source_key, value in state.items():
        if not isinstance(source_key, str) or not isinstance(value, Tensor):
            raise TypeError("Checkpoint state must map string keys to tensors")
        if value.device.type != "cpu":
            raise ValueError(
                f"Checkpoint tensor {source_key!r} is on {value.device}; "
                "expected cpu"
            )
        if value.layout != torch.strided:
            raise ValueError(
                f"Checkpoint tensor {source_key!r} has layout {value.layout}; "
                f"expected {torch.strided}"
            )

        if source_key == "prediction_head.alphas":
            if task != "regression":
                raise ValueError(
                    "Classification checkpoint contains quantile alphas"
                )
            _validate_alphas(value)
            has_alphas = True
            continue

        if source_key.startswith(("y_encoder.lin.", "icl_y_encoder.lin.")):
            if task == "classification":
                if source_key.endswith(".bias"):
                    continue
                prefix = source_key.removesuffix(".lin.weight")
                target = (
                    "row_embedding.y_emb.weight"
                    if prefix == "y_encoder"
                    else "icl_block.y_emb.weight"
                )
                _insert(
                    out,
                    target,
                    _classification_embedding(state, prefix),
                    source_key,
                )
                continue

            if source_key.endswith(".bias"):
                raise ValueError(
                    f"Regression checkpoint has unexpected bias {source_key!r}"
                )
            target = (
                "row_embedding.y_lin.weight"
                if source_key.startswith("y_encoder.")
                else "icl_block.y_lin.weight"
            )
            _insert(out, target, value, source_key)
            continue

        for target_key in _map_key(source_key):
            _insert(out, target_key, value, source_key)

    if task == "regression" and not has_alphas:
        raise ValueError(
            "Regression checkpoint is missing 'prediction_head.alphas'"
        )
    return out


def _validate_args(args: object, *, task: Task) -> None:
    if not isinstance(args, Mapping):
        raise TypeError("Checkpoint 'args' must be a mapping")

    actual = dict(args)
    expected = _EXPECTED_ARGS_BY_TASK[task]
    if not isinstance(actual.get("grad_checkpoint"), bool):
        raise ValueError(
            "Unsupported Kumo Tabular checkpoint argument: "
            "'grad_checkpoint' must be bool"
        )
    missing = sorted(key for key in expected if key not in actual)
    unexpected = sorted(repr(key) for key in actual if key not in expected)
    changed = sorted(
        key
        for key, expected_value in expected.items()
        if key != "grad_checkpoint"
        and key in actual
        and actual[key] != expected_value
    )
    if not missing and not unexpected and not changed:
        return
    raise ValueError(
        "Unsupported Kumo Tabular checkpoint arguments: "
        f"missing={missing}, unexpected={unexpected}, changed={changed}"
    )


def _validate_checkpoint_schema(checkpoint: Mapping[object, object]) -> None:
    expected = {"model", "args", "step", "config"}
    missing = sorted(key for key in expected if key not in checkpoint)
    unexpected = sorted(repr(key) for key in checkpoint if key not in expected)
    if missing or unexpected:
        raise ValueError(
            "Unsupported Kumo Tabular checkpoint schema: "
            f"missing={missing}, unexpected={unexpected}"
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


def _load_checkpoint(
    model: _Model,
    checkpoint_path: str | Path,
    *,
    task: Task,
    device: torch.device | str | None,
) -> _Model:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    _validate_checkpoint_schema(checkpoint)
    _validate_args(checkpoint["args"], task=task)

    raw_state = checkpoint["model"]
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint 'model' must be a mapping")
    state = _remap_checkpoint(
        cast(Mapping[str, Tensor], raw_state),
        task=task,
    )
    _validate_state(state, model.state_dict())
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device).eval()
