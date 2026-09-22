# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool the predictions of the query rows that one group holds.

A label-per-group task carries one label for a whole group of rows. A
customer has many transaction rows, and the label says whether the
customer commits fraud. A prediction that differs between the rows of
one customer is wrong on some of them, so the mean over the group
removes that error. For a convex loss the mean never scores worse than
the rows it replaces.

BeyondArena names the label type and the group column, so nothing has
to infer them. The default pipeline drops the group column, because
the splits are group aware and an id of the query never appears in the
context. :class:`KumoGroupPreprocessing` keeps a copy under a reserved
name, and the adapter removes it before the model reads the table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tabarena.benchmark.preprocessing import TabArenaModelAgnosticPreprocessing
from tabarena.benchmark.task.metadata import GroupLabelTypes

#: The column that carries the group id to the adapter.
GROUP_ID_COLUMN = "__kumo_group_id__"


def pool_by_group(
    probabilities: np.ndarray,
    group_id: pd.Series,
) -> np.ndarray:
    """Give every row of one group the mean probability of the group."""
    frame = pd.DataFrame(probabilities)
    pooled = frame.groupby(group_id.to_numpy(), sort=False).transform("mean")
    return pooled.to_numpy()


class KumoGroupPreprocessing(TabArenaModelAgnosticPreprocessing):
    """Keep a copy of the group column of a label-per-group task.

    A label-per-sample task holds one label per row, so the mean over
    its groups would be wrong. It gets no column, and the pool then
    cannot reach it.
    """

    def __init__(
        self,
        group_cols: str | list[str] | None = None,
        group_labels: GroupLabelTypes | None = None,
        group_time_on: str | None = None,
        **kwargs: Any,
    ) -> None:
        # ``build_feature_generator`` forwards a group parameter only
        # when the signature names it. Therefore ``group_time_on`` has
        # to appear here, although it only goes to the base class.
        self.group_columns: list[str] = []
        if group_labels == GroupLabelTypes.PER_GROUP and group_cols:
            # A task names one group column or several of them.
            self.group_columns = (
                [group_cols]
                if isinstance(group_cols, str)
                else list(group_cols)
            )
        super().__init__(
            group_cols=group_cols,
            group_labels=group_labels,
            group_time_on=group_time_on,
            **kwargs,
        )

    def _attach(self, out: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        """Add the group id to the output of the default pipeline."""
        columns = [name for name in self.group_columns if name in x.columns]
        if not columns:
            return out
        keys = x[columns[0]].astype(str)
        for name in columns[1:]:
            # The separator never appears in a value, so two different
            # groups cannot build the same key.
            keys = keys + "\x1f" + x[name].astype(str)
        # A number survives the later steps that a string does not.
        codes = pd.factorize(keys)[0]
        return out.assign(**{GROUP_ID_COLUMN: codes.astype("float64")})

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self._attach(super().fit_transform(X, y=y, **kwargs), X)

    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self._attach(super().transform(X, **kwargs), X)
