# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolate the prediction and latency effects of using SDM RMSNorm."""

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from chronos import Chronos2Pipeline
from chronos.chronos2.layers import Chronos2LayerNorm

from benchmark.timeseries.chronos2.parity import (
    CASES,
    REVISION,
    difference,
    inputs,
    measure,
)
from sdm.models.chronos2 import Chronos2
from sdm.models.chronos2.ckpt import remap_ckpt
from sdm.nn import RMSNorm


def original_norm(model: Chronos2) -> Chronos2:
    model = copy.deepcopy(model)
    replacements: dict[int, Chronos2LayerNorm] = {}
    # Preserve the shared query/key normalization objects. The pinned upstream
    # class implements the same arithmetic as the removed custom LayerNorm.
    for name, module in list(model.named_modules(remove_duplicate=False)):
        if isinstance(module, RMSNorm):
            if id(module) not in replacements:
                norm = Chronos2LayerNorm(module.weight.numel(), eps=1e-6)
                norm.weight = module.weight
                replacements[id(module)] = norm
            model.set_submodule(name, replacements[id(module)])
    return model.eval()


@torch.inference_mode()
def compare(
    baseline: Chronos2,
    plain: Chronos2,
    upstream: Chronos2Pipeline,
    case: str,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dictionaries, tables, *_ = inputs(case, "cuda")
    horizon = CASES[case][2]

    def autocast() -> Any:
        if mode == "autocast_float16":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def previous() -> torch.Tensor:
        with autocast():
            return baseline(*tables).numerical

    def candidate() -> torch.Tensor:
        with autocast():
            return plain(*tables).numerical

    with autocast():
        reference = (
            torch.stack(
                upstream.predict(
                    dictionaries, prediction_length=horizon, batch_size=10000
                )
            )
            .permute(0, 3, 1, 2)
            .flatten(-2)
        )
    a, b = previous(), candidate()
    latency = measure((previous, candidate), args.warmup, args.repeats)
    return {
        "mode": mode,
        "case": case,
        "shape": CASES[case],
        "plain_vs_previous": difference(a, b),
        "previous_vs_upstream": difference(reference, a),
        "plain_vs_upstream": difference(reference, b),
        "previous_latency": latency[0],
        "plain_latency": latency[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    result: dict[str, Any] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": REVISION,
        "source": "1224fafa91e1ecd6674f5204066b159e40b5e4a1",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    for mode in ("float32", "float16", "autocast_float16"):
        dtype = torch.float16 if mode == "float16" else torch.float32
        upstream = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            revision=REVISION,
            device_map="cuda",
            dtype=dtype,
        )
        plain = Chronos2(pretrained=False, device="meta")
        plain.model.load_state_dict(
            remap_ckpt(upstream.model.state_dict()), strict=True, assign=True
        )
        baseline = original_norm(plain)
        for case in args.cases:
            row = compare(baseline, plain, upstream, case, mode, args)
            print(json.dumps(row), flush=True)
            result["cases"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        del upstream, plain, baseline


if __name__ == "__main__":
    main()
