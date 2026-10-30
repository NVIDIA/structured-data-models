# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the pinned public Chronos-2 pipeline and SDM on identical inputs."""

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from chronos import Chronos2Pipeline

from sdm import Stype, TableTensor
from sdm.models.chronos2 import Chronos2
from sdm.models.chronos2.ckpt import remap_ckpt
from sdm.processing import Standardize

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"

CASES = {
    "univariate": (1, 95, 31, 1, 0, 0),
    "batch_univariate": (16, 512, 96, 1, 0, 0),
    "multivariate": (2, 512, 96, 3, 0, 0),
    "covariates": (2, 255, 65, 2, 1, 2),
    "missing": (2, 95, 33, 2, 1, 1),
    "all_missing": (2, 47, 17, 1, 0, 0),
    "constant": (1, 127, 32, 2, 0, 0),
    "one_step": (1, 1, 1, 1, 0, 0),
    "truncated_context": (1, 8301, 17, 1, 0, 1),
    "long_horizon": (1, 95, 1041, 1, 1, 1),
    "three_chunks": (1, 47, 2051, 1, 0, 0),
}


def inputs(name: str, device: str) -> tuple:
    batch, length, horizon, targets, past, known = CASES[name]
    total = targets + past + known
    index = torch.arange(batch * total * (length + horizon)).float()
    values = (index * 0.07).sin() + (index * 0.013).cos() * 0.5
    values = values.reshape(batch, total, length + horizon)
    context = values[..., :length].clone()
    future = values[..., length:].clone()
    future[:, : targets + past] = torch.nan
    if name == "missing":
        context[..., :21] = torch.nan
        context[..., 25::7] = torch.nan
        context[0, 0] = torch.nan
        future[..., ::5] = torch.nan
    elif name == "all_missing":
        context[:] = torch.nan
    elif name == "constant":
        context[:, 0] = 0
        context[:, 1] = 100
    elif name == "truncated_context":
        context[..., :30] = 1e6
    dictionaries = []
    for row in range(batch):
        dictionaries.append(
            {
                "target": context[row, :targets].numpy(),
                "past_covariates": {
                    f"past{i}": context[row, targets + i].numpy()
                    for i in range(past)
                }
                | {
                    f"known{i}": context[row, targets + past + i].numpy()
                    for i in range(known)
                },
                "future_covariates": {
                    f"known{i}": future[row, targets + past + i].numpy()
                    for i in range(known)
                },
            }
        )
    features = context[:, targets:].transpose(-1, -2).to(device)
    query = future[:, targets + past :].transpose(-1, -2).to(device)
    names = [f"past{i}" for i in range(past)] + [
        f"known{i}" for i in range(known)
    ]
    x = TableTensor.from_tensor(features.unsqueeze(0), columns=names)
    y = TableTensor.from_tensor(
        context[:, :targets].transpose(-1, -2).to(device).unsqueeze(0),
        columns=[f"target{i}" for i in range(targets)],
    )
    q = TableTensor.from_tensor(query.unsqueeze(0), columns=names[past:])
    groups = torch.arange(batch, device=device).repeat_interleave(total)
    return (
        dictionaries,
        (x, y, q),
        context.flatten(0, 1).to(device),
        future.flatten(0, 1).to(device),
        groups,
    )


def difference(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().cpu(), b.float().cpu()
    if a.shape != b.shape:
        raise RuntimeError(f"Output shapes differ: {a.shape} vs {b.shape}")
    delta = (a - b).abs()
    scale = a.square().mean().sqrt().clamp_min(1e-6)
    return {
        "max_abs": delta.max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_rmse": (delta.square().mean().sqrt() / scale).item(),
        "max_series_relative_rmse": (
            delta.square().mean(dim=-2).sqrt()
            / a.square().mean(dim=-2).sqrt().clamp_min(1e-6)
        )
        .max()
        .item(),
        "finite": bool(a.isfinite().all() and b.isfinite().all()),
    }


def measure(
    calls: tuple[Callable, Callable], warmup: int, repeats: int
) -> list[dict[str, float]]:
    for _ in range(warmup):
        for call in calls:
            call()
    walls: list[list[float]] = [[], []]
    events: list[list[float]] = [[], []]
    for repeat in range(repeats):
        for index in range(2) if repeat % 2 == 0 else reversed(range(2)):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            start.record()
            calls[index]()
            end.record()
            torch.cuda.synchronize()
            walls[index].append((time.perf_counter() - tick) * 1000)
            events[index].append(start.elapsed_time(end))
    return [
        {
            "wall_ms": statistics.median(wall),
            "cuda_ms": statistics.median(event),
            "wall_p10_ms": sorted(wall)[len(wall) // 10],
            "wall_p90_ms": sorted(wall)[
                min(len(wall) - 1, 9 * len(wall) // 10)
            ],
        }
        for wall, event in zip(walls, events)
    ]


@torch.inference_mode()
def run_case(
    upstream: Chronos2Pipeline,
    model: Chronos2,
    case: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dictionaries, tables, context, future, groups = inputs(case, "cuda")
    _, _, horizon, targets, _, _ = CASES[case]

    def reference() -> torch.Tensor:
        out = upstream.predict(
            dictionaries,
            prediction_length=horizon,
            batch_size=10000,
        )
        return torch.stack(out).permute(0, 3, 1, 2).flatten(-2)

    def candidate() -> torch.Tensor:
        return model(*tables).numerical

    a, b = reference(), candidate()
    public_diff = difference(a, b)
    expected_names = [
        f"target{i}__q{round(q * 100)}"
        for i in range(targets)
        for q in upstream.quantiles
    ]
    if list(model(*tables).columns[Stype.numerical]) != expected_names:
        raise RuntimeError("Quantile column order differs")
    public_time = measure((reference, candidate), args.warmup, args.repeats)
    # The public upstream API includes CPU input preparation and a CPU
    # result copy. Compare numerical execution separately on CUDA.
    patches = min((horizon + 15) // 16, 64)

    def reference_core() -> torch.Tensor:
        return upstream._predict_step(
            context=context,
            group_ids=groups,
            future_covariates=future,
            num_output_patches=patches,
        )

    def candidate_core() -> torch.Tensor:
        processor = Standardize()
        retained = context[..., -model.model.context_length :]
        table = TableTensor.from_tensor(retained.T)
        x = processor.fit_transform(table).numerical.T.asinh()
        f = processor.transform(
            TableTensor.from_tensor(future.T)
        ).numerical.T.asinh()
        batch, _, _, targets, past, known = CASES[case]
        variates = targets + past + known
        out = (
            model.model._predict(
                x.unflatten(0, (batch, variates)),
                f.unflatten(0, (batch, variates)),
                patches,
            )
            .flatten(0, 1)
            .sinh()
        )
        # Put the quantile axis first so fitted [1, S] statistics broadcast.
        table = TableTensor.from_tensor(out.permute(1, 2, 0))
        return processor.inverse_transform(table).numerical.permute(2, 0, 1)

    core_diff = difference(reference_core(), candidate_core())
    core_time = measure(
        (reference_core, candidate_core), args.warmup, args.repeats
    )
    return {
        "dtype": str(next(model.parameters()).dtype),
        "case": case,
        "shape": CASES[case],
        "public_difference": public_diff,
        "core_difference": core_diff,
        "public_upstream": public_time[0],
        "public_sdm": public_time[1],
        "core_upstream": core_time[0],
        "core_sdm": core_time[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16"])
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    results: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": version("transformers"),
        "chronos_forecasting": version("chronos-forecasting"),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "recipe": repr(Chronos2.default_recipe()),
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "checkpoint": REVISION,
        "source": "1224fafa91e1ecd6674f5204066b159e40b5e4a1",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    for name in args.dtypes:
        dtype = getattr(torch, name)
        upstream = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            revision=REVISION,
            device_map="cuda",
            dtype=dtype,
        )
        model = Chronos2(device="cuda")
        if dtype != torch.float32:
            state = remap_ckpt(upstream.model.state_dict())
            model.model.load_state_dict(state, strict=True, assign=True)
        for case in args.cases:
            row = run_case(upstream, model, case, args)
            print(json.dumps(row), flush=True)
            results["cases"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        del upstream, model


if __name__ == "__main__":
    main()
