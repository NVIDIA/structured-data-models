"""Compare warmed CPU prediction with generated versus native LayerNorm."""
# ruff: noqa: D103, T201

import argparse
import copy
import json
import statistics
import time
import types
from pathlib import Path

import native_layer_norm_patch
import torch

from sdm.models import KumoRelational

# Use per-instance replacements so the baseline remains unchanged.
torch.nn.LayerNorm.forward = native_layer_norm_patch._original_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    arm = torch.load(args.data, weights_only=False)["arms"][1]
    model = KumoRelational(
        task="classification", pretrained=False, device="cpu"
    )
    model.models["classification"].load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    )
    query = arm["queries"][args.query]
    result = {
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "query": args.query,
        "cold": [],
        "timings": [],
    }
    with torch.inference_mode():
        model.fit(
            arm["context"],
            arm["context_target"],
            arm["related_context"],
            generator=torch.Generator().manual_seed(123),
            num_estimators=1,
            num_hops=arm["num_hops"],
        )
        expected = model.predict(
            query["x"], query["related"]
        ).numerical.clone()
        generated = copy.deepcopy(model)
        native = copy.deepcopy(model)
        for module in native.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.forward = types.MethodType(
                    native_layer_norm_patch._forward, module
                )
        for compiled_model in [generated, native]:
            compiled_model.models["classification"].compile(
                backend="inductor", fullgraph=True, dynamic=True
            )
        models = {
            "eager": model,
            "generated_norm": generated,
            "native_norm": native,
        }
        for name, m in models.items():
            start = time.perf_counter()
            actual = m.predict(query["x"], query["related"]).numerical
            result["cold"].append(
                {
                    "path": name,
                    "seconds": time.perf_counter() - start,
                    "max_abs_error": (actual - expected).abs().max().item(),
                    "parity": torch.allclose(
                        actual, expected, atol=1e-5, rtol=1e-4
                    ),
                }
            )
        names = list(models)
        for repeat in range(12):
            for name in names[repeat % 3 :] + names[: repeat % 3]:
                start = time.perf_counter()
                actual = (
                    models[name]
                    .predict(query["x"], query["related"])
                    .numerical
                )
                elapsed = time.perf_counter() - start
                if repeat >= 3:
                    result["timings"].append(
                        {
                            "path": name,
                            "repeat": repeat - 3,
                            "ms": elapsed * 1000,
                            "max_abs_error": (actual - expected)
                            .abs()
                            .max()
                            .item(),
                            "parity": torch.allclose(
                                actual, expected, atol=1e-5, rtol=1e-4
                            ),
                        }
                    )
        result["summary"] = [
            {
                "path": name,
                "median_ms": statistics.median(
                    x["ms"] for x in result["timings"] if x["path"] == name
                ),
                "min_ms": min(
                    x["ms"] for x in result["timings"] if x["path"] == name
                ),
                "max_ms": max(
                    x["ms"] for x in result["timings"] if x["path"] == name
                ),
            }
            for name in names
        ]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
