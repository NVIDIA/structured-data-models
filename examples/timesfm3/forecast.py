# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_context = sdm.TableTensor.from_tensor(
    torch.tensor(
        [
            [18.0, 1.00],
            [20.0, 0.95],
            [float("nan"), 0.90],
            [23.0, 0.90],
            [25.0, 0.85],
            [24.0, 0.80],
        ],
        device=device,
    ),
    columns=["observed_weather", "planned_price"],
)
y_context = sdm.TableTensor.from_tensor(
    torch.tensor(
        [
            [42.0, 18.0],
            [45.0, 20.0],
            [51.0, 23.0],
            [57.0, 25.0],
            [63.0, 28.0],
            [61.0, 27.0],
        ],
        device=device,
    ),
    columns=["ice_cream", "cold_drinks"],
)
x_query = sdm.TableTensor.from_tensor(
    torch.tensor([[0.80], [0.75], [0.75]], device=device),
    columns=["planned_price"],
)

# The first run prompts for acceptance of Google's non-commercial license.
model = sdm.models.TimesFM3(device=device)
model.fit(x_context, y_context)
forecast = model.predict(x_query)

print(forecast.columns[sdm.Stype.numerical])
print(forecast.numerical)
