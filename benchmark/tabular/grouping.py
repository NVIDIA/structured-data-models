# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the group id of a grouped task, in the form its label type needs.

A grouped task names the column that holds the group of a row, and the
default pipeline of BeyondArena drops that column. Two features want
it back. Each one applies to one label type, so no task gets both.

A label-per-group task carries one label for a whole group of rows. A
customer has many transaction rows, and the label says whether the
customer commits fraud. A prediction that differs between the rows of
one customer is wrong on some of them, so the mean over the group
removes that error. For a convex loss the mean never scores worse than
the rows it replaces. The id is no feature here. The pipeline copies
it into a reserved column, and the adapter removes that column before
the model reads the table.

A label-per-sample task carries one label per row, and the drop of the
group column is the only grouped step it gets. Here the id stays as a
feature. A grouped split shares no group with the context, so a map
that is fitted on the context gives every query row a value the
context never held. A hash needs no fitted map, so a new group lands
inside the range that the context holds, and the rows of one group
still share a value.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from tabarena.benchmark.preprocessing import TabArenaModelAgnosticPreprocessing
from tabarena.benchmark.task.metadata import GroupLabelTypes

#: The column that carries the group id to the adapter.
GROUP_ID_COLUMN = "__kumo_group_id__"

#: The number of values that the hash of a group id can take.
BUCKETS = 64


def _bucket(key: str) -> float:
    """Map a group id to one of :data:`BUCKETS` values between 0 and 1."""
    # ``hash`` changes between processes, so this uses a digest.
    digest = hashlib.md5(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS / BUCKETS


def pool_by_group(
    probabilities: np.ndarray,
    group_id: pd.Series,
) -> np.ndarray:
    """Give every row of one group the mean probability of the group."""
    frame = pd.DataFrame(probabilities)
    pooled = frame.groupby(group_id.to_numpy(), sort=False).transform("mean")
    return pooled.to_numpy()


class KumoGroupPreprocessing(TabArenaModelAgnosticPreprocessing):
    """Keep the group column, which the default pipeline drops.

    The label type picks the form. A label-per-group task gets the id
    in :data:`GROUP_ID_COLUMN`, which is no feature. A label-per-sample
    task gets it as a hash under its own name, which is a feature. A
    task that the flags leave out reaches the base class unchanged.
    """

    def __init__(
        self,
        group_cols: str | list[str] | None = None,
        group_labels: GroupLabelTypes | None = None,
        group_time_on: str | None = None,
        group_pooling: bool = False,
        group_id: bool = False,
        **kwargs: Any,
    ) -> None:
        # ``build_feature_generator`` forwards a group parameter only
        # when the signature names it. Therefore ``group_time_on`` has
        # to appear here, although it only goes to the base class.
        # A task names one group column or several of them.
        names = (
            [group_cols]
            if isinstance(group_cols, str)
            else list(group_cols or [])
        )
        self.group_columns: list[str] = []
        self.feature_columns: list[str] = []
        if group_pooling and group_labels == GroupLabelTypes.PER_GROUP:
            self.group_columns = names
        if group_id and group_labels == GroupLabelTypes.PER_SAMPLE:
            self.feature_columns = names
            # The base class adds the step that drops the group
            # columns only when it receives them.
            group_cols = group_labels = None
        super().__init__(
            group_cols=group_cols,
            group_labels=group_labels,
            group_time_on=group_time_on,
            **kwargs,
        )

    def _encode(self, x: pd.DataFrame) -> pd.DataFrame:
        """Replace a group id by a hash, which a new group survives."""
        columns = [name for name in self.feature_columns if name in x.columns]
        if not columns:
            return x
        codes = {
            name: x[name].astype(str).map(_bucket).astype("float64")
            for name in columns
        }
        return x.assign(**codes)

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
        # The hash runs first, so the id reaches the steps of the base
        # class as a feature does.
        out = super().fit_transform(self._encode(X), y=y, **kwargs)
        return self._attach(out, X)

    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self._attach(super().transform(self._encode(X), **kwargs), X)
