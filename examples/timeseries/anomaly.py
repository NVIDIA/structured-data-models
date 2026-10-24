# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time = torch.arange(200, device=device, dtype=torch.float32)
x = torch.stack([torch.sin(time / 12), torch.cos(time / 20)], dim=-1)
x[140:150] += 3  # Add anomalies.

model = sdm.models.KumoAnomaly(device=device)
out = model(x)

# Higher MAE scores indicate more likely anomalies:
print("Reconstruction Error (MAE):")
print(out["mae"].numerical[130:160, 0])

# SCS flags anomalies using adaptive segments of the preprocessed series:
print("Anomaly Labels (SCS):")
print(out["scs"].categorical.decode()[130:160, 0])

# MACS flags anomalies using variance-weighted score bounds:
print("Anomaly Labels (MACS):")
print(out["macs"].categorical.decode()[130:160, 0])
