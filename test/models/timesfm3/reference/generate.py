# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate TimesFM-3 reference fixtures with pinned Google code."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

UPSTREAM_REVISION = "e31dadd84cb26bd5153fde6687502b8312e918fb"
WEIGHT_RECIPE = {
    "offset_step": 3,
    "modulus": 19,
    "center": 9,
    "divisor": 32,
}
PUBLIC_CONFIG: dict[str, Any] = {
    "input_patch_len": 2,
    "output_patch_len": 4,
    "quantiles": [0.1, 0.5, 0.9],
    "residual_block_config": {
        "hidden_dims": 8,
        "output_dims": 8,
        "use_bias": False,
        "activation": "relu",
    },
    "transformer_config": {
        "num_layers": 2,
        "transformer": {
            "model_dims": 8,
            "hidden_dims": 12,
            "num_heads": 2,
            "attention_norm": "rms",
            "feedforward_norm": "rms",
            "qk_norm": "rms",
            "use_rope_seq": True,
            "use_rope_var": False,
            "use_bias": False,
            "ff_activation": "relu",
            "deterministic": True,
        },
    },
    "use_variate_attention": True,
    "use_stitching": True,
    "use_linear_detrending": True,
    "use_iterative_cpm_revin": True,
    "use_frozen_running_stats": False,
}
INTERNAL_CONFIG: dict[str, Any] = {
    "input_patch_len": 2,
    "output_patch_len": 4,
    "quantiles": [0.5],
    "residual_block_config": {
        "hidden_dims": 4,
        "output_dims": 4,
        "use_bias": False,
        "activation": "relu",
    },
    "transformer_config": {
        "num_layers": 1,
        "transformer": {
            "model_dims": 4,
            "hidden_dims": 6,
            "num_heads": 2,
            "attention_norm": "rms",
            "feedforward_norm": "rms",
            "qk_norm": "rms",
            "use_rope_seq": True,
            "use_rope_var": False,
            "use_bias": False,
            "ff_activation": "relu",
            "deterministic": True,
        },
    },
    "use_variate_attention": False,
    "use_stitching": True,
    "use_linear_detrending": True,
    "use_frozen_running_stats": False,
}
PUBLIC_INPUTS: dict[str, Any] = {
    "targets": {
        "columns": ["target_a", "target_b"],
        "context": [
            [1.0, 10.0],
            [2.0, 8.0],
            [None, 7.0],
            [5.0, None],
            [8.0, 3.0],
        ],
    },
    "past_only": {
        "columns": ["past"],
        "context": [[3.0], [None], [4.0], [1.0], [5.0]],
    },
    "future_known": {
        "columns": ["known_a", "known_b"],
        "context": [
            [5.0, 2.0],
            [6.0, 3.0],
            [4.0, None],
            [9.0, 5.0],
            [7.0, 8.0],
        ],
        "query": [[8.0, 13.0], [None, 21.0], [10.0, 34.0]],
    },
}
FORWARD_INPUTS: dict[str, Any] = {
    "values": [[[[1.0, 3.0], [0.0, 0.0], [0.0, 0.0]]]],
    "masks": [[[[False, False], [True, True], [True, True]]]],
    "patch_is_target": [[[True, True, True]]],
    "patch_cpm_masks": [None, [[False, True, True]]],
}
DECODE_INPUTS: dict[str, Any] = {
    "target": [[[1.0, 2.0, 5.0, 3.0, 8.0]]],
    "past_only_covariates": [[[3.0, 1.0, 4.0, 1.0, 5.0]]],
    "past_future_covariates": [[[5.0, 6.0, 4.0, 9.0, 7.0, 8.0, 6.0, 10.0]]],
    "mask": [[False, False, False, True, False]],
    "past_only_mask": [[[False, True, False, False, False]]],
    "past_future_mask": [
        [[False, False, False, False, False, False, True, False]]
    ],
    "horizon": 99,
}
NO_STITCHING_INPUTS: dict[str, Any] = {
    "target": [[[1.0, 3.0, 2.0, 5.0]]],
    "target_mask": [[[False, False, True, False]]],
    "horizon": 5,
}


def _matrix(values: list[list[float | None]]) -> torch.Tensor:
    return torch.tensor(
        [
            [float("nan") if value is None else value for value in row]
            for row in values
        ],
        dtype=torch.float32,
    )


def _fill_state(model: torch.nn.Module) -> list[dict[str, Any]]:
    state = model.state_dict()
    entries = []
    for index, key in enumerate(sorted(state)):
        tensor = state[key]
        values = (
            (
                torch.arange(tensor.numel()).reshape(tensor.shape)
                + WEIGHT_RECIPE["offset_step"] * index
            )
            % WEIGHT_RECIPE["modulus"]
            - WEIGHT_RECIPE["center"]
        ) / WEIGHT_RECIPE["divisor"]
        state[key] = values.to(dtype=tensor.dtype)
        entries.append({"key": key, "shape": list(tensor.shape)})
    model.load_state_dict(state, strict=True)
    return entries


def _upstream_revision(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _public_fixture(upstream: Any) -> dict[str, Any]:
    model = upstream.TimesFM3Torch(**PUBLIC_CONFIG).eval()
    state = _fill_state(model)
    targets = _matrix(PUBLIC_INPUTS["targets"]["context"]).T.unsqueeze(0)
    past_only = _matrix(PUBLIC_INPUTS["past_only"]["context"]).T.unsqueeze(0)
    future_context = _matrix(PUBLIC_INPUTS["future_known"]["context"])
    future_query = _matrix(PUBLIC_INPUTS["future_known"]["query"])
    future_known = torch.cat([future_context, future_query]).T.unsqueeze(0)

    with torch.inference_mode():
        forecasts = model.decode(
            targets.nan_to_num(),
            past_only_covariates=past_only.nan_to_num(),
            past_future_covariates=future_known.nan_to_num(),
            target_mask=~targets.isfinite(),
            past_only_mask=~past_only.isfinite(),
            past_future_mask=~future_known.isfinite(),
        )

    num_targets = targets.shape[1]
    expected = forecasts[:, :num_targets].permute(0, 2, 1, 3)
    expected = expected.reshape(
        expected.shape[1], num_targets * len(PUBLIC_CONFIG["quantiles"])
    )
    output_columns = [
        f"{column}__q{quantile * 100:g}"
        for column in PUBLIC_INPUTS["targets"]["columns"]
        for quantile in PUBLIC_CONFIG["quantiles"]
    ]
    return {
        "config": PUBLIC_CONFIG,
        "state": state,
        "inputs": PUBLIC_INPUTS,
        "expected": {
            "columns": output_columns,
            "values": expected.tolist(),
        },
    }


def _internal_fixture(upstream: Any) -> dict[str, Any]:
    model = upstream.TimesFM3Torch(**INTERNAL_CONFIG).eval()
    state = _fill_state(model)
    values = torch.tensor(FORWARD_INPUTS["values"])
    masks = torch.tensor(FORWARD_INPUTS["masks"])
    patch_is_target = torch.tensor(FORWARD_INPUTS["patch_is_target"])
    forward_outputs = []
    with torch.inference_mode():
        for patch_cpm_mask in FORWARD_INPUTS["patch_cpm_masks"]:
            cpm_mask = (
                None
                if patch_cpm_mask is None
                else torch.tensor(patch_cpm_mask)
            )
            logits = model(
                {
                    "values": values,
                    "masks": masks,
                    "patch_is_target": patch_is_target,
                },
                patch_cpm_mask=cpm_mask,
            )["logits"]
            forward_outputs.append(logits.tolist())

        decode = model.decode(
            torch.tensor(DECODE_INPUTS["target"]),
            horizon=DECODE_INPUTS["horizon"],
            past_only_covariates=torch.tensor(
                DECODE_INPUTS["past_only_covariates"]
            ),
            past_future_covariates=torch.tensor(
                DECODE_INPUTS["past_future_covariates"]
            ),
            mask=torch.tensor(DECODE_INPUTS["mask"]),
            past_only_mask=torch.tensor(DECODE_INPUTS["past_only_mask"]),
            past_future_mask=torch.tensor(DECODE_INPUTS["past_future_mask"]),
        )

    no_stitching_config = {
        **INTERNAL_CONFIG,
        "use_stitching": False,
        "use_linear_detrending": False,
        "use_frozen_running_stats": True,
    }
    no_stitching_model = upstream.TimesFM3Torch(**no_stitching_config).eval()
    _fill_state(no_stitching_model)
    with torch.inference_mode():
        no_stitching = no_stitching_model.decode(
            torch.tensor(NO_STITCHING_INPUTS["target"]),
            horizon=NO_STITCHING_INPUTS["horizon"],
            target_mask=torch.tensor(NO_STITCHING_INPUTS["target_mask"]),
        )

    return {
        "config": INTERNAL_CONFIG,
        "state": state,
        "forward": {
            "inputs": FORWARD_INPUTS,
            "outputs": forward_outputs,
        },
        "decode": {
            "inputs": DECODE_INPUTS,
            "output": decode.tolist(),
        },
        "decode_without_stitching": {
            "config": no_stitching_config,
            "inputs": NO_STITCHING_INPUTS,
            "output": no_stitching.tolist(),
        },
    }


def generate(checkout: Path) -> dict[str, Any]:
    """Generate the fixtures using an exact Google TimesFM checkout."""
    revision = _upstream_revision(checkout)
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(
            f"Expected upstream revision {UPSTREAM_REVISION}, got {revision}."
        )

    sys.path.insert(0, str(checkout / "src"))
    upstream = importlib.import_module("timesfm3.torch.model")
    return {
        "upstream": {
            "repository": "https://github.com/google-research/timesfm",
            "revision": revision,
        },
        "weight_recipe": WEIGHT_RECIPE,
        "public": _public_fixture(upstream),
        "internal": _internal_fixture(upstream),
    }


def main() -> None:
    """Generate and write the reference fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout",
        type=Path,
        help="Path to the pinned google-research/timesfm checkout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("golden.json"),
    )
    args = parser.parse_args()

    fixture = generate(args.checkout.resolve())
    args.output.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
