# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time = torch.arange(200, device=device, dtype=torch.float32)
x = torch.stack([torch.sin(time / 12), torch.cos(time / 20)], dim=-1)
x[140:149] += 3  # Add anomalies.

model = sdm.models.KumoAnomaly(device=device)

# Higher reconstruction error scores indicate more likely anomalies:
score = model(x).numerical.squeeze(-1)
print(score[131:158])
