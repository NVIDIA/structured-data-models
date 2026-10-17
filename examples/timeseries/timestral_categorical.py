# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch

import sdm

# Forecast daily sales conditioned on future event labels and past footfall.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time = torch.arange(96).float()
events = ["weekend" if i % 7 >= 5 else "weekday" for i in range(110)]
context = sdm.TableTensor.from_pandas(
    pd.DataFrame(
        {
            "sales": 20 + 3 * (time / 7).sin(),
            "footfall": 100 + time,
            "event": pd.Categorical(events[:96]),
        }
    ),
    stypes={
        "sales": "numerical",
        "footfall": "numerical",
        "event": "categorical",
    },
    device=device,
)
query = sdm.TableTensor.from_pandas(
    pd.DataFrame({"event": pd.Categorical(events[96:])}),
    stypes={"event": "categorical"},
    device=device,
)
model = sdm.models.Timestral(model_size="5M", device=device)
forecast = model(
    x_context=context.drop_columns("sales"),
    y_context=context["sales"],
    x_query=query,
)
print(forecast["sales__q50"].numerical.squeeze(-1).cpu())
