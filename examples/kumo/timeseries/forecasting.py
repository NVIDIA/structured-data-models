# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pretrained Kumo forecasting on a multivariate synthetic history.

Run from the repository root:
    uv run python examples/kumo/timeseries/forecasting.py

The first run downloads the 1.4 GB cross-channel checkpoint. This verifies
inference wiring; a synthetic series does not establish forecasting accuracy.
"""

import torch

import sdm


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sdm.models.KumoForecasting(device=device)
    time = torch.arange(512, dtype=torch.float32, device=device)
    history = torch.stack([(time / 12).sin(), (time / 24).cos()], dim=-1)
    target = sdm.TableTensor.from_tensor(
        history, columns=["signal_a", "signal_b"]
    )
    context = torch.empty(512, 0, device=device)
    query = torch.empty(72, 0, device=device)

    prediction = model(context, target, query)
    model.fit(context, target)
    cached = model.predict(query)
    torch.testing.assert_close(cached.numerical, prediction.numerical)
    assert prediction.shape == (72, 2)
    assert prediction.numerical.isfinite().all()
    print(f"Forecast shape: {tuple(prediction.shape)}")
    print(f"Target columns: {prediction.columns['numerical']}")
    print(prediction.numerical[:5])


if __name__ == "__main__":
    main()
