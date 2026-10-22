# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark public SDM and upstream Timestral-5M forecast latency."""

import argparse
import json
import statistics
from importlib.metadata import distribution
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from benchmark.forecasting.timestral_parity import (
    SOURCE_REVISION,
    WEIGHT_REVISION,
    make_case,
)
from sdm import TableTensor
from sdm.models import Timestral


def main() -> None:
    from timestral import TimestralForecaster  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    installed = json.loads(
        distribution("timestral").read_text("direct_url.json") or "{}"
    )
    assert installed.get("vcs_info", {}).get("commit_id") == SOURCE_REVISION
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = Timestral(device=device, accept_license=True)
    upstream = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M",
        revision=WEIGHT_REVISION,
        device=device,
    )
    cases = [
        ("univariate", 128, 96),
        ("univariate", 1024, 96),
        ("multivariate", 1024, 96),
        ("numeric", 1024, 96),
        ("categorical", 1024, 96),
        ("long_horizon", 79, 1030),
    ]
    results = []
    for case, context_length, horizon in cases:
        x, y, query, covariates = make_case(
            case,
            args.device,
            context_length=context_length,
            prediction_length=horizon,
        )
        target = y.numerical.transpose(-2, -1)

        def sdm_forecast(
            x: TableTensor = x,
            y: TableTensor = y,
            query: TableTensor = query,
        ) -> torch.Tensor:
            return model(x, y, query, num_estimators=1).numerical

        def upstream_forecast(
            target: torch.Tensor = target,
            query: TableTensor = query,
            covariates: dict = covariates,
        ) -> torch.Tensor:
            return (
                upstream.forecast(target, query.size(-2), **covariates)
                .quantiles.transpose(-3, -2)
                .flatten(-2)
            )

        # Check exactly the paths that will be timed, including their defaults.
        actual, expected = sdm_forecast(), upstream_forecast()
        tol = 1e-4 if device.type == "cuda" else 2e-5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        for _ in range(args.warmup):
            sdm_forecast()
            upstream_forecast()
        calls = {"sdm": sdm_forecast, "upstream": upstream_forecast}
        samples: dict[str, list[float]] = {name: [] for name in calls}
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic thermal/order bias.
            order = (
                ("sdm", "upstream") if repeat % 2 == 0 else ("upstream", "sdm")
            )
            for name in order:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = perf_counter()
                calls[name]()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples[name].append(1000 * (perf_counter() - start))
        row: dict[str, Any] = {
            "case": case,
            "batch": target.size(0),
            "targets": target.size(1),
            "context": target.size(2),
            "horizon": query.size(-2),
            "median_ms": {
                name: statistics.median(values)
                for name, values in samples.items()
            },
            "p25_p75_ms": {
                name: [
                    statistics.quantiles(values, n=4)[0],
                    statistics.quantiles(values, n=4)[2],
                ]
                for name, values in samples.items()
            },
            "samples_ms": samples,
        }
        row["upstream_over_sdm"] = (
            row["median_ms"]["upstream"] / row["median_ms"]["sdm"]
        )
        results.append(row)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key != "samples_ms"
                }
            ),
            flush=True,
        )
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "source_revision": SOURCE_REVISION,
        "weight_revision": WEIGHT_REVISION,
        "timing": (
            "Per-call wall time, CUDA synchronized before and after each "
            "call; alternating implementation order"
        ),
        "results": results,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
