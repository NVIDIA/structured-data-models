# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare public Timestral forecasts against the pinned upstream 5M model.

Install git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff
then run python -m benchmark.forecasting.timestral_parity --device cpu
or --device cuda. This downloads the 5M weights under their Modified MIT terms.
"""

import argparse
import json
from importlib.metadata import distribution
from time import perf_counter

import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import Timestral

SOURCE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
WEIGHT_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"


def make_case(
    name: str,
    device: str,
    *,
    context_length: int = 79,
    prediction_length: int = 65,
) -> tuple[TableTensor, TableTensor, TableTensor, dict]:
    t, h, v, batch = context_length, prediction_length, 1, 1
    if name == "multivariate":
        v = 3
    if name == "batched":
        batch = 2
    if name == "long_horizon":
        h = 1030
    if name == "context_cap":
        t, h = 32775, 1
    if name == "single_patch":
        t, h = 7, 1
    time = torch.arange(t + h, device=device).float()
    target = (time[:t] / 9).sin().expand(batch, v, t).clone()
    target += torch.arange(v, device=device)[None, :, None] * 0.2
    target += torch.arange(batch, device=device)[:, None, None] * 0.3
    if name == "positive":
        target = target.abs() + 1
    if name == "constant":
        target.fill_(2)
    if name in {"missing", "categorical_missing"}:
        target[..., :8] = torch.nan
        target[..., 12:15] = torch.nan
    if name == "all_missing":
        target.fill_(torch.nan)
    reference = {}
    context = TableTensor(size=(batch, t), device=device)
    query = TableTensor(size=(batch, h), device=device)
    if name in {
        "numeric",
        "indicator",
        "categorical",
        "categorical_multi",
        "categorical_missing",
    }:
        if name == "categorical_multi":
            target = target.expand(batch, 2, t).clone()
            v = 2
        past = (time[:t] / 13).cos().expand(batch, 1, t).clone()
        known = (time / 7).sin().expand(batch, 1, t + h).clone()
        if name == "indicator":
            known = (known > 0).float()
        if name.startswith("categorical"):
            past = (time[:t].long() % 3).expand(batch, 1, t).clone()
            known = (time.long() % 4).expand(batch, 1, t + h).clone()
            past[..., 3:6] = -1
            known[..., 3:6] = -1
            known[..., t + 4 :] = 4  # Category first seen in the horizon.

            def table(codes: torch.Tensor, columns: list[str]) -> TableTensor:
                return TableTensor(
                    columns={Stype.categorical: columns},
                    categorical=CategoricalTensor(
                        codes.transpose(-2, -1),
                        [torch.arange(5, device=device) for _ in columns],
                    ),
                )

            context = table(
                torch.cat([known[..., :t], past], dim=1), ["known", "past"]
            )
            query = table(known[..., t:], ["known"])
            reference = {
                "categorical_past_covariates": past.float().masked_fill(
                    past < 0, torch.nan
                ),
                "categorical_known_covariates": known.float().masked_fill(
                    known < 0, torch.nan
                ),
            }
        else:
            past[..., 4:6] = torch.nan
            context = TableTensor.from_tensor(
                torch.cat([known[..., :t], past], dim=1).transpose(-2, -1),
                columns=["known", "past"],
            )
            query = TableTensor.from_tensor(
                known[..., t:].transpose(-2, -1), columns=["known"]
            )
            reference = {"past_covariates": past, "known_covariates": known}
    targets = TableTensor.from_tensor(
        target.transpose(-2, -1), columns=[f"target{i}" for i in range(v)]
    )
    return context, targets, query, reference


def main() -> None:
    from timestral import TimestralForecaster  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    installed = json.loads(
        distribution("timestral").read_text("direct_url.json") or "{}"
    )
    assert installed.get("vcs_info", {}).get("commit_id") == SOURCE_REVISION
    torch.set_num_threads(2)
    model = Timestral(device=args.device, accept_license=True)
    upstream = TimestralForecaster.from_pretrained(
        "mistralai/Timestral-5M",
        revision=WEIGHT_REVISION,
        device=args.device,
    )
    cases = [
        "single_patch",
        "univariate",
        "multivariate",
        "batched",
        "missing",
        "all_missing",
        "constant",
        "positive",
        "numeric",
        "indicator",
        "categorical",
        "categorical_multi",
        "categorical_missing",
        "long_horizon",
        "context_cap",
    ]
    report = {}
    for name in cases:
        x, y, query, covariates = make_case(name, args.device)
        target = y.numerical.transpose(-2, -1)
        expected = (
            upstream.forecast(target, query.size(-2), **covariates)
            .quantiles.transpose(-3, -2)
            .flatten(-2)
        )
        actual = model(x, y, query, num_estimators=1).numerical
        tol = 1e-4 if args.device.startswith("cuda") else 2e-5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        model.fit(
            x,
            y,
            context_only_columns=["past"]
            if "past" in x.column_names
            else None,
            num_estimators=1,
        )
        torch.testing.assert_close(model.predict(query).numerical, actual)
        report[name] = {
            "max_absolute_error": (actual - expected).abs().max().item(),
            "max_scaled_error": (
                (actual - expected).abs() / (1 + expected.abs())
            )
            .max()
            .item(),
        }
        print(name, report[name], flush=True)
    x, y, query, _ = make_case("numeric", args.device)
    for _ in range(3):
        model(x, y, query, num_estimators=1)
    if args.device.startswith("cuda"):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(args.repeats):
            model(x, y, query, num_estimators=1)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / args.repeats
    else:
        start_time = perf_counter()
        for _ in range(args.repeats):
            model(x, y, query, num_estimators=1)
        elapsed = 1000 * (perf_counter() - start_time) / args.repeats
    print(
        json.dumps(
            {
                "device": args.device,
                "torch": torch.__version__,
                "source_revision": SOURCE_REVISION,
                "weight_revision": WEIGHT_REVISION,
                "cases": report,
                "numeric_forward_ms": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
