# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark full-graph TabICLv2 classification on CUDA.

Run from the repository root, for example::

    python -m benchmark.tabular.tabiclv2_compile --classes 26
    python -m benchmark.tabular.tabiclv2_compile --pretrained --classes 26

First-call timing includes compilation; disk compiler caches may be warm.
Warm timings synchronize each measurement.
The ICL-only mode isolates hierarchical routing from row embedding costs.
Architecture options apply to synthetic models; pretrained uses its defaults.
Results identify whether the class count selects hierarchical routing.
"""

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
from torch import Tensor

from sdm.cache import Cache
from sdm.models import TabICLv2
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.model import _TabICLv2


def measure(fn: Callable[[], Tensor], repeats: int) -> dict[str, float]:
    r"""Measure synchronized latency and peak allocated memory after warmup."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "peak_extra_mib": (torch.cuda.max_memory_allocated() - baseline)
        / 2**20,
    }


def main() -> None:
    r"""Compare eager and compiled outputs, then report CUDA measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=26)
    parser.add_argument("--capacity", type=int, default=10)
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--test-rows", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--icl-only", action="store_true")
    model_group.add_argument(
        "--pretrained",
        action="store_true",
        help="Use the published classifier with its default architecture",
    )
    parser.add_argument("--eager-only", action="store_true")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="float32"
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    dtype = getattr(torch, args.dtype)
    capacity = args.capacity
    model: torch.nn.Module
    if args.pretrained:
        model = (
            TabICLv2(task="classification", device="cuda")
            .models["classification"]
            .to(dtype=dtype)
        )
        capacity = model.icl_block.num_classes
    elif args.icl_only:
        model = ICLBlock(
            num_classes=args.capacity,
            out_channels=args.capacity,
            channels=args.channels,
            num_layers=args.layers,
            num_heads=args.heads,
            norm_bias=True,
            temperature=0.9,
            device="cuda",
            dtype=dtype,
        )
    else:
        model = _TabICLv2(
            num_classes=args.capacity,
            num_quantiles=3,
            channels=args.channels,
            num_embedding_layers=1,
            num_embedding_heads=4,
            num_inducing_points=8,
            group_size=1,
            num_readout_tokens=1,
            num_icl_layers=args.layers,
            num_icl_heads=args.heads,
            device="cuda",
            dtype=dtype,
        )
    model.eval()
    if not args.pretrained:
        # Exercise zero-initialized residual branches in synthetic models.
        for parameter in model.parameters():
            if (
                parameter.ndim > 1
                and parameter.detach().count_nonzero().item() == 0
            ):
                torch.nn.init.normal_(parameter, std=0.02)
    x = torch.randn(
        args.batch_size,
        args.train_rows + args.test_rows,
        args.channels if args.icl_only else args.features,
        device="cuda",
        dtype=dtype,
    )
    y = (torch.arange(args.train_rows, device="cuda") % args.classes).expand(
        args.batch_size, -1
    )
    result: dict[str, object] = {
        "arguments": vars(args),
        "capacity": capacity,
        "hierarchical": args.classes > capacity,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "checkpoint": "jingang/TabICL/tabicl-classifier-v2-20260212.ckpt"
        if args.pretrained
        else None,
    }
    with torch.no_grad():
        expected = model(x.clone(), y, num_classes=args.classes).clone()
        cache = None
        if args.cached:
            cache = Cache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(
                x[:, : args.train_rows].clone(),
                y,
                num_classes=args.classes,
                cache=cache,
            )
            torch.cuda.synchronize()
            result["fit_seconds"] = time.perf_counter() - start
            result["cache_mib"] = cache.size() / 2**20
            cache.freeze()
            x = x[:, args.train_rows :]
            y = y[:, :0]

        def eager() -> Tensor:
            return model(x.clone(), y, num_classes=args.classes, cache=cache)

        tolerance = 0.02 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(
            eager(), expected, atol=tolerance, rtol=tolerance
        )
        result["eager"] = measure(
            eager,
            repeats=args.repeats,
        )
        if not args.eager_only:
            compiled = torch.compile(
                model, fullgraph=True, dynamic=args.dynamic
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual = compiled(
                x.clone(), y, num_classes=args.classes, cache=cache
            ).clone()
            torch.cuda.synchronize()
            result["cold_seconds"] = time.perf_counter() - start
            torch.testing.assert_close(
                actual, expected, atol=tolerance, rtol=tolerance
            )
            finite = expected.isfinite()
            result["max_logit_error"] = (
                (actual[finite] - expected[finite]).abs().max().item()
            )
            result["max_probability_error"] = (
                (
                    (actual / 0.9).softmax(dim=-1)
                    - (expected / 0.9).softmax(dim=-1)
                )
                .abs()
                .max()
                .item()
            )
            result["prediction_agreement"] = (
                (actual.argmax(dim=-1) == expected.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )
            result["compiled"] = measure(
                lambda: compiled(
                    x.clone(), y, num_classes=args.classes, cache=cache
                ),
                repeats=args.repeats,
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
