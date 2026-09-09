"""Generic matured training-label history, not V11 raw-event aggregates."""

import numpy as np
import pandas as pd


class LabelHistory:
    """Prepare a reusable history using only the labels supplied to fit.

    Args:
        train: Fit's training frame; never append prediction-time labels.
        entity_col: Entity identifier column.
        time_col: Label-window start or query cutoff column.
        target_col: Training target column.
        time_horizon: Time until a historical label is fully observed.
        n_lags: Number of prior values and ages; the shared V12 candidate is 2.

    Using the full training history does not increase the model's separately
    sampled 10,000 context rows per estimator. This optional feature candidate
    has no task-name dispatch and is not equivalent to V11's event aggregates.
    """

    def __init__(
        self,
        train: pd.DataFrame,
        *,
        entity_col: str,
        time_col: str,
        target_col: str,
        time_horizon: pd.Timedelta,
        n_lags: int = 2,
    ) -> None:
        self.entity_col = entity_col
        self.time_col = time_col
        self.time_horizon = pd.Timedelta(time_horizon)
        if self.time_horizon < pd.Timedelta(0):
            raise ValueError(
                "Historical labels cannot have a negative horizon"
            )
        self.n_lags = n_lags
        self.columns = tuple(
            name
            for lag in range(1, n_lags + 1)
            for name in (f"history_target_{lag}", f"history_age_days_{lag}")
        )
        history = train[[entity_col, time_col, target_col]].copy()
        history.columns = ["entity", "time", "target"]
        history["time"] = pd.to_datetime(history["time"])
        history = history.dropna().sort_values("time", kind="stable")
        grouped = history.groupby("entity", sort=False)
        prepared = history[["entity", "time"]].copy()
        prepared["entity"] = prepared["entity"].astype(object)
        prepared["ready"] = prepared["time"] + self.time_horizon
        for lag in range(1, n_lags + 1):
            prepared[f"value_{lag}"] = grouped["target"].shift(lag - 1)
            prepared[f"time_{lag}"] = grouped["time"].shift(lag - 1)
        self.history = prepared.drop(columns="time")

    def transform(self, queries: pd.DataFrame) -> pd.DataFrame:
        """Return numerical history features in the original query row order.

        Only query entities and cutoffs are read. A source label is eligible
        when its full window has ended by the cutoff and its own anchor is
        strictly earlier. Missing entities/history/timestamps produce NaNs.
        """
        output = pd.DataFrame(
            np.nan, index=queries.index, columns=self.columns, dtype=np.float32
        )
        anchors = queries[[self.entity_col, self.time_col]].copy()
        anchors.columns = ["entity", "cutoff"]
        anchors["cutoff"] = pd.to_datetime(anchors["cutoff"])
        anchors["position"] = np.arange(len(anchors))
        anchors = anchors.dropna().sort_values("cutoff", kind="stable")
        if anchors.empty or self.history.empty:
            return output
        anchors["entity"] = anchors["entity"].astype(object)
        matched = pd.merge_asof(
            anchors,
            self.history,
            left_on="cutoff",
            right_on="ready",
            by="entity",
            direction="backward",
            allow_exact_matches=self.time_horizon > pd.Timedelta(0),
        )
        positions = matched["position"].to_numpy()
        for lag in range(1, self.n_lags + 1):
            output.iloc[positions, 2 * (lag - 1)] = matched[
                f"value_{lag}"
            ].to_numpy(dtype=np.float32)
            output.iloc[positions, 2 * (lag - 1) + 1] = (
                (matched["cutoff"] - matched[f"time_{lag}"])
                / pd.Timedelta(days=1)
            ).to_numpy(dtype=np.float32)
        return output
