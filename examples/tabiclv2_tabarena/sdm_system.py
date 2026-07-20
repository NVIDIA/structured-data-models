"""SDM-native TabArena system adapter for local TabICLv2 evaluation."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sdm import Stype, StypeLike, TableTensor, infer_stypes
from sdm.models import TabICLv2
from tabarena.benchmark.exec_models.external import ExternalSystemModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer
    from tabarena.benchmark.task.metadata import ValidationMetadata


@dataclass(frozen=True)
class FeatureSchema:
    """The SDM-owned feature contract fitted from one training frame."""

    columns: tuple[str, ...]
    stypes: dict[str, Stype]
    id_columns: tuple[str, ...]


class SDMTabICLv2System(ExternalSystemModel):
    """Run local TabICLv2 while SDM owns feature and target preprocessing."""

    def __init__(self, *, num_estimators: int = 8, **kwargs: object) -> None:
        if num_estimators < 1:
            raise ValueError("'num_estimators' must be positive")
        super().__init__(**kwargs)
        self.num_estimators = num_estimators

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ) -> SDMTabICLv2System:
        """Fit TabICLv2 on raw TabArena data through SDM's recipe."""
        del (
            eval_metric,
            validation_metadata,
            num_cpus,
            memory_limit,
            time_limit,
            random_state,
        )
        if problem_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(
                f"Unsupported TabArena problem type '{problem_type}'"
            )

        self._schema = _fit_feature_schema(X)
        X = _align_features(X, schema=self._schema)
        self._device = _resolve_device(num_gpus=num_gpus)
        self._problem_type = problem_type
        self._target_name = target_name or "__target__"
        self._target_stype = (
            Stype.numerical
            if problem_type == "regression"
            else Stype.categorical
        )
        if problem_type != "regression":
            self._class_labels_by_key = _class_labels_by_key(y)

        self.model = TabICLv2(device=self._device)
        self.model.fit(
            x=_table_from_frame(
                X, stypes=self._schema.stypes, device=self._device
            ),
            y=_table_from_series(
                y,
                name=self._target_name,
                stype=self._target_stype,
                device=self._device,
            ),
            num_estimators=self.num_estimators,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Return indexed point predictions for a regression task."""
        if self._problem_type != "regression":
            raise RuntimeError("Classification tasks require '_predict_proba'")
        values = self._prediction_values(X)
        return pd.Series(
            values.mean(axis=-1), index=X.index, name=self._target_name
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class-labelled probabilities without label cleaning.

        AutoGluon never transforms labels on this system path.
        """
        if self._problem_type == "regression":
            raise RuntimeError("Regression tasks require '_predict'")
        prediction = self._predict_table(X)
        values = prediction.numerical.float().cpu().numpy()
        class_keys = prediction.columns[Stype.numerical]
        labels = _labels_from_prediction_columns(
            class_keys,
            labels_by_key=self._class_labels_by_key,
        )
        return pd.DataFrame(
            values,
            index=X.index,
            columns=np.asarray(labels, dtype=object),
        )

    def _prediction_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return numerical TabICLv2 output values for a raw query frame."""
        return self._predict_table(X).numerical.float().cpu().numpy()

    def _predict_table(self, X: pd.DataFrame) -> TableTensor:
        """Run cached TabICLv2 inference after SDM-owned schema validation."""
        if not hasattr(self, "model"):
            raise RuntimeError(
                "SDMTabICLv2System must be fitted before prediction"
            )
        X = _align_features(X, schema=self._schema)
        return self.model.predict(
            _table_from_frame(
                X, stypes=self._schema.stypes, device=self._device
            )
        )

    def cleanup(self) -> None:
        """Release local TabICLv2 caches and CUDA allocations after a task."""
        if hasattr(self, "model"):
            self.model.clear()
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _fit_feature_schema(frame: pd.DataFrame) -> FeatureSchema:
    """Infer and validate the SDM-owned feature contract from training data."""
    stypes = {
        name: Stype(stype) for name, stype in infer_stypes(frame).items()
    }
    datetime_columns = [
        name for name, stype in stypes.items() if stype == Stype.datetime
    ]
    if datetime_columns:
        columns = ", ".join(repr(column) for column in datetime_columns)
        raise ValueError(
            "SDM-native TabICLv2 does not yet support datetime columns: "
            f"{columns}. Use the AutoGluon-compatible mode or add an SDM "
            "datetime recipe before running this task."
        )

    id_columns = tuple(
        name for name, stype in stypes.items() if stype == Stype.id
    )
    columns = tuple(name for name in frame.columns if name not in id_columns)
    if not columns:
        raise ValueError(
            "SDM-native TabICLv2 requires at least one non-ID feature"
        )

    return FeatureSchema(
        columns=columns,
        stypes={name: stypes[name] for name in columns},
        id_columns=id_columns,
    )


def _align_features(
    frame: pd.DataFrame, *, schema: FeatureSchema
) -> pd.DataFrame:
    """Validate raw query schema and return the fitted non-ID column order."""
    expected = set(schema.columns)
    allowed = expected | set(schema.id_columns)
    actual = set(frame.columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - allowed)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise ValueError(
            "Feature schema mismatch (" + "; ".join(details) + ")"
        )

    actual_stypes = {
        name: Stype(stype)
        for name, stype in infer_stypes(
            frame.loc[:, list(schema.columns)]
        ).items()
    }
    changed = {
        name: (schema.stypes[name], actual_stypes[name])
        for name in schema.columns
        if actual_stypes[name] != schema.stypes[name]
    }
    if changed:
        details = ", ".join(
            f"{name!r}: {expected.value} -> {actual.value}"
            for name, (expected, actual) in changed.items()
        )
        raise ValueError(
            f"Feature semantic types changed since fit ({details})"
        )
    return frame.loc[:, list(schema.columns)]


def _class_labels_by_key(y: pd.Series) -> dict[str, object]:
    """Map SDM's string output identifiers back to original pandas labels."""
    if y.isna().any():
        raise ValueError(
            "Classification targets must not contain missing values"
        )

    labels: dict[str, object] = {}
    for label in pd.unique(y):
        key = str(label)
        if key in labels:
            raise ValueError(
                "Classification labels have ambiguous string representations: "
                f"{labels[key]!r} and {label!r} both map to {key!r}"
            )
        labels[key] = label
    return labels


def _labels_from_prediction_columns(
    columns: tuple[str, ...],
    *,
    labels_by_key: dict[str, object],
) -> list[object]:
    """Resolve TabICLv2 prediction columns to original class labels."""
    missing = [column for column in columns if column not in labels_by_key]
    if missing:
        raise RuntimeError(
            "TabICLv2 returned class columns absent from the fitted target "
            f"labels: {missing}"
        )
    return [labels_by_key[column] for column in columns]


def _resolve_device(*, num_gpus: int | None) -> torch.device:
    """Select CPU or CUDA from TabArena's requested resource count."""
    if num_gpus is None or num_gpus <= 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("TabArena requested a GPU but CUDA is unavailable")
    return torch.device("cuda")


def _table_from_frame(
    frame: pd.DataFrame,
    *,
    stypes: Mapping[str, StypeLike],
    device: torch.device,
) -> TableTensor:
    """Convert a raw feature frame through the declared SDM semantic types."""
    return TableTensor.from_pandas(df=frame, stypes=stypes, device=device)


def _table_from_series(
    series: pd.Series,
    *,
    name: str,
    stype: Stype,
    device: torch.device,
) -> TableTensor:
    """Convert a raw target series to a single-column SDM table."""
    return TableTensor.from_pandas(
        df=series.rename(name).to_frame(),
        stypes={name: stype},
        device=device,
    )
