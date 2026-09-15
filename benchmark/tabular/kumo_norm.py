# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native and custom normalization in complete KumoTabular forwards.

Run each shape in a fresh process, on an otherwise idle CUDA GPU::

    python -m benchmark.tabular.kumo_norm --checkpoint classifier.pt \
        --size small --rows 1024 --columns 100

Use the matching small/large KumoTabular checkpoint. Timings exclude data
preprocessing, transfers and compilation startup. Compiled arms use native
normalization, as intended by the custom kernels' dispatch policy.
"""

import argparse
import copy
import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor, nn

from sdm.models.kumo.tabular.ckpt import remap_ckpt
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular
from sdm.nn._rmsnorm_for_linear import _RMSNormForLinear
from sdm.nn._rope_rmsnorm import _RoPERMSNorm


def native_normalization(model: nn.Module) -> None:
    """Replace the two normalization wrappers with native operations."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, _RMSNormForLinear):
                norm = nn.RMSNorm(
                    torch.Size(child.normalized_shape),
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                    device="cuda",
                )
                norm.load_state_dict(child.state_dict())
                setattr(parent, name, norm.eval())
            elif isinstance(child, _RoPERMSNorm):
                setattr(parent, name, nn.Sequential(*child).eval())


def difference(actual: Tensor, reference: Tensor) -> dict:
    """Report numerical differences without treating compilation as exact."""
    return {
        "exact": torch.equal(actual, reference),
        "max_abs": (actual.float() - reference.float()).abs().max().item(),
        "finite": actual.isfinite().all().item(),
        "close": torch.allclose(actual, reference, atol=2e-3, rtol=2e-3),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    """Check outputs and measure randomized, synchronized forward calls."""
    torch.manual_seed(8162)
    random.seed(917)
    custom = _KumoTabular(
        num_classes=10, num_quantiles=0, **MODEL_KWARGS[args.size]
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    custom.load_state_dict(
        remap_ckpt(
            checkpoint["model"],
            is_classifier=True,
            num_layers=MODEL_KWARGS[args.size]["num_embedding_layers"],
        )
    )
    custom = custom.cuda().eval()
    native = copy.deepcopy(custom)
    native_normalization(native)
    models: dict[str, Callable[..., Tensor]] = {
        "native_eager": native,
        "custom_eager": custom,
    }
    for name, model in list(models.items()):
        models[name.replace("eager", "compiled")] = torch.compile(
            model,
            fullgraph=False,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )

    x = torch.randn(args.batch, args.rows, args.columns, device="cuda")
    categorical = (
        torch.arange(args.columns, device="cuda")[None, :].expand(
            args.batch, -1
        )
        % 4
        == 0
    )
    x = torch.where(categorical[:, None, :], (x.abs() * 4).round(), x)
    x[:, 3, 1] = float("nan")
    y = torch.randint(0, 3, (args.batch, args.rows * 3 // 4), device="cuda")
    with torch.autocast("cuda", dtype=getattr(torch, args.dtype)):
        reference = native(x, y, categorical).clone()
        outputs = {
            name: model(x, y, categorical).clone()
            for name, model in models.items()
        }
        checks = {
            name: difference(output, reference)
            for name, output in outputs.items()
        }
        checks["compiled_comparison"] = difference(
            outputs["custom_compiled"], outputs["native_compiled"]
        )
        torch.testing.assert_close(
            outputs["custom_eager"], reference, atol=0, rtol=0
        )
        for model in models.values():
            for _ in range(3):
                model(x, y, categorical)
        samples = {name: [] for name in models}
        for _ in range(args.repeats):
            names = list(models)
            random.shuffle(names)
            for name in names:
                torch.cuda.synchronize()
                start = time.perf_counter()
                models[name](x, y, categorical)
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "size": args.size,
                "shape": [args.batch, args.rows, args.columns],
                "dtype": args.dtype,
                "checks": checks,
                "samples_ms": samples,
                "median_ms": {
                    name: statistics.median(v) for name, v in samples.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--size", choices=["small", "large"], required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=100)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument("--repeats", type=int, default=9)
    run(parser.parse_args())
