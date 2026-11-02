# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import relbench


def add_lag_target(
    df: pd.DataFrame,
    task: relbench.base.EntityTask,
    num_lags: int,
) -> pd.DataFrame:
    r"""Add lag target features to the task table."""
    df = df.copy()
    right = df.sort_values(task.time_col)
    left = pd.DataFrame(
        {
            task.entity_col: df[task.entity_col],
            "__lookup_time__": df[task.time_col],
            "__row__": range(len(df)),
        }
    )
    for lag in range(1, num_lags + 1):
        column = f"{task.target_col}_lag_{lag}"
        if task.task_type == relbench.base.TaskType.REGRESSION:
            df[column] = float("NaN")
        else:
            df[column] = pd.Series(pd.NA, index=df.index, dtype="object")

        merged = pd.merge_asof(
            left.sort_values("__lookup_time__"),
            right,
            by=task.entity_col,
            left_on="__lookup_time__",
            right_on=task.time_col,
            direction="backward",
            allow_exact_matches=False,
        )
        mask = merged[task.time_col].notna()
        rows = merged.loc[mask, "__row__"].to_numpy()
        df.loc[rows, column] = merged.loc[mask, task.target_col].to_numpy()

        # Find the next lag strictly before the previous observation.
        left = merged.loc[mask, [task.entity_col, "__row__", task.time_col]]
        left = left.rename(columns={task.time_col: "__lookup_time__"})

    return df
