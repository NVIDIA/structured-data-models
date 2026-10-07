"""Check diagnostic LayerNorm dtype behavior against native CUDA LayerNorm."""
# ruff: noqa: T201

import argparse
import json
from pathlib import Path

import native_layer_norm_patch  # noqa: F401
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
results = []
with torch.inference_mode():
    for input_dtype, weight_dtype in [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ]:
        for autocast in [False, True]:
            torch._dynamo.reset()
            model = torch.nn.LayerNorm(128, device="cuda", dtype=weight_dtype)
            model.weight.uniform_(0.5, 1.5)
            model.bias.normal_()
            x = torch.randn(
                3, 17, 128, device="cuda", dtype=input_dtype
            ).transpose(0, 1)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=autocast
            ):
                expected = model(x)
                actual = torch.compile(model, backend="eager", fullgraph=True)(
                    x
                )
            results.append(
                {
                    "input_dtype": str(input_dtype),
                    "weight_dtype": str(weight_dtype),
                    "autocast_bf16": autocast,
                    "expected_dtype": str(expected.dtype),
                    "actual_dtype": str(actual.dtype),
                    "equal": torch.equal(actual, expected),
                    "max_abs_error": (actual - expected).abs().max().item(),
                }
            )
Path(args.output).write_text(
    json.dumps(
        {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(results), flush=True)
