# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score anomalies in an ordered multivariate time series."""

import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time = torch.arange(200, device=device, dtype=torch.float32)
values = torch.stack([torch.sin(time / 12), torch.cos(time / 20)], dim=-1)
values[140:150] += 3
series = sdm.TableTensor.from_tensor(values)
model = sdm.models.KumoAnomaly(device=device)

# Preprocessing fits on this series; reconstruction uses each whole window.
# Scores measure reconstruction error, not anomaly probabilities.
scores = model(
    x=series,
    num_samples=3,
    num_steps=20,
    generator=torch.Generator(device=device).manual_seed(42),
)
print(scores.numerical.squeeze(-1))
