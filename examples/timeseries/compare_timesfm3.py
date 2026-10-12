# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from time import perf_counter

import torch
from timesfm3 import TimesFM3Forecaster

import sdm
from sdm.models.timesfm3 import TimesFM3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare univariate TimesFM3 forecasts with upstream."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()

    time = torch.arange(args.context_length, dtype=torch.float32)
    history = time.div(9).sin() + 0.3 * time.div(23).cos()
    y_context = sdm.TableTensor.from_tensor(
        history.unsqueeze(-1).to(args.device), columns=["target"]
    )
    # No covariates; empty feature tensors still specify the row counts.
    x_context = torch.empty(args.context_length, 0, device=args.device)
    x_query = torch.empty(args.horizon, 0, device=args.device)

    # Load SDM first so its license prompt precedes any weight download.
    model = TimesFM3(device=args.device, accept_license=args.accept_license)
    official = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch",
        device=args.device,
    )
    # These upstream refinements are not implemented in SDM yet.
    official.model.use_linear_detrending = False
    official.model.use_iterative_cpm_revin = False

    t = perf_counter()
    reference = official.predict(
        context=history.numpy(),
        horizon=args.horizon,
        return_quantiles=True,
        sort_quantiles=True,
    )
    print("official", perf_counter() - t)
    assert reference.quantiles is not None
    expected = torch.from_numpy(reference.quantiles)
    t = perf_counter()
    actual = model(x_context, y_context, x_query).numerical.cpu()
    print("actual", perf_counter() - t)
    # Sort only the nine quantiles of this single target, as upstream does.
    actual = actual.sort(dim=-1).values

    error = actual - expected
    print(f"Max absolute error: {error.abs().max().item():.6g}")
    print(f"RMSE: {error.square().mean().sqrt().item():.6g}")
    print("Step     Official q50       SDM q50")
    for step in range(min(args.horizon, 8)):
        print(
            f"{step:4d}     {expected[step, 4]:12.6f}  {actual[step, 4]:12.6f}"
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("All nine quantiles match within rtol=1e-4, atol=1e-4.")


if __name__ == "__main__":
    main()
