# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch

import sdm

url = "https://autogluon.s3.amazonaws.com/datasets/timeseries"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sdm.models.TimesFM3(device=device)

print("# Univariate Forecasting")  ############################################
# 414 items ("item_id"), each with up to 960 observations:
df = pd.read_csv(f"{url}/m4_hourly/train.csv")
df = df.sort_values(["item_id", "timestamp"])  # Left pad with NaNs:
df["step"] = df.groupby("item_id").cumcount(ascending=False)
df = df.pivot(index="item_id", columns="step", values="target").iloc[:, ::-1]

tensor = torch.from_numpy(df.to_numpy(dtype="float32", copy=True)).to(device)
table = sdm.TableTensor.from_tensor(tensor.unsqueeze(-1), columns=["target"])

with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    out = model(
        x_context=table.drop_columns("target"),
        y_context=table["target"],
        # Forecast 24 observations per item:
        x_query=sdm.TableTensor(size=(table.size(0), 24), device=device),
        num_estimators=2,
    )["target__q50"].numerical.squeeze(-1)  # Median prediction
print(f"24H forecast for item '{df.index[0]}':")
print(out[0].cpu().numpy(), end="\n\n")

print("# Forecasting with Covariates")  #######################################
# Hourly energy prices for the next day using historical prices and day-ahead
# forecasts of load and renewable energy generation
context_df = pd.read_parquet(f"{url}/electricity_price/train.parquet")
query_df = pd.read_parquet(f"{url}/electricity_price/test.parquet")
stypes = {
    "Ampirion Load Forecast": "numerical",
    "PV+Wind Forecast": "numerical",
    "target": "numerical",
}
context = sdm.TableTensor.from_pandas(context_df, stypes, device=device)
query = sdm.TableTensor.from_pandas(query_df, stypes, device=device)

with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    out = model(
        x_context=context.drop_columns("target"),
        y_context=context["target"],
        x_query=query.drop_columns("target"),
        num_estimators=2,
    )["target__q50"].numerical.squeeze(-1)  # Median prediction
print("24H energy price forecast for region 'DE':")
print(out.cpu().numpy(), end="\n\n")

print("# Forecasting with Past-only Covariates")  #############################
# Forecast next quarter's weekly store sales using historical sales, historical
# customer footfall, and known covariates indicating store operation, promotion
# periods, and holidays.
context_df = pd.read_parquet(f"{url}/retail_sales/train.parquet")
context_df = context_df.sort_values(["id", "timestamp"])
query_df = pd.read_parquet(f"{url}/retail_sales/test.parquet")
query_df = query_df.sort_values(["id", "timestamp"])
stypes = {
    "Sales": "numerical",
    "Open": "numerical",
    "Promo": "numerical",
    "SchoolHoliday": "numerical",
    "StateHoliday": "numerical",
}
context = sdm.TableTensor.from_pandas(
    context_df,
    stypes=stypes | {"Customers": "numerical"},  # Add past-only covariate.
    device=device,
).view(1115, 120, -1)  # 1115 stores + 120 weeks
query = sdm.TableTensor.from_pandas(
    query_df,
    stypes,
    device=device,
).view(1115, 13, -1)  # 1115 stores + 13 weeks

with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    out = model(
        x_context=context.drop_columns("Sales"),
        y_context=context["Sales"],  # type: ignore
        x_query=query.drop_columns("Sales"),
        num_estimators=2,
    )["Sales__q50"].numerical.squeeze(-1)  # Median prediction
print("Weekly sales forecast for item '1':")
print(out[0].cpu().numpy())
