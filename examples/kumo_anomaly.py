# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score and flag anomalies with reference-only feature preprocessing."""

import torch

import sdm
from sdm.models import KumoAnomaly
from sdm.models.kumo.timeseries.anomaly import MACSThreshold, SCSThreshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time = torch.arange(600, device=device, dtype=torch.float32)
values = torch.stack([torch.sin(time / 12), torch.cos(time / 20)], dim=-1)
values[450:460] += 3
series = sdm.TableTensor.from_tensor(values)
model = KumoAnomaly(device=device)

# Reference rows fit PCA/min-max only; model weights are already pretrained.
# Reconstruction uses future rows within each query window, so this is an
# offline detector, not a causal streaming evaluation.
result = model(
    x=series[300:],
    reference=series[:300],
    num_samples=3,
    num_steps=20,
    generator=torch.Generator(device=device).manual_seed(42),
)
print(result.scores)
# These SDK rules use the scored query itself, including its global score
# percentile. Reference-only feature scaling does not make them causal or
# reference-calibrated. The corrected solver can change scores and decisions.
anomalies_scs = SCSThreshold()(result.scores, result.target)
anomalies_macs = MACSThreshold()(result.scores)
print("SCS anomaly rows:", anomalies_scs.nonzero().flatten() + 300)
print("MACS anomaly rows:", anomalies_macs.nonzero().flatten() + 300)
# The booleans are detector decisions, not ground truth or probabilities.
# Validate detection quality and threshold settings on your own labeled data.
