# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forecast an incomplete history with KumoForecastMissingness."""

import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sdm.models.KumoForecastMissingness(device=device)

# Ordered historical observations only. Keep NaNs instead of zero-imputing.
time = torch.arange(300, device=device, dtype=torch.float32)
values = torch.stack([time.sin(), (time / 12).cos()], dim=-1)
values[240:272, 0] = torch.nan
values[::5, 1] = torch.nan
history = sdm.TableTensor.from_tensor(values, columns=["target", "HUFL"])

# Named auxiliary channels use checkpoint identities. Other names require
# channel_alignment="positional", which assigns identities by input order.
prediction = model(
    x_context=history.drop_columns("target"),
    y_context=history["target"],
    x_query=sdm.TableTensor(size=(48,), device=device),
)
print(prediction.numerical)  # [48, 1], in original target units.
