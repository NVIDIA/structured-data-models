"""Run the repository-local TabICLv2 model on TabArena."""

# Quick smoke run:
# uv run --group example-tabarena python examples/tabiclv2_tabarena.py \
#   --output-root "$(mktemp -d)" --mode sdm-native --subset lite \
#   --datasets blood-transfusion-service-center --num-estimators 1 \
#   --num-cpus 1 --num-gpus 0

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.models import AbstractModel
from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2
from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata.collection import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator, SystemConfigGenerator

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer
    from tabarena.benchmark.task.metadata import ValidationMetadata


class SDMTabICLv2Model(AbstractModel):
    """Expose local :class:`sdm.models.TabICLv2` through AutoGluon."""

    ag_key = "SDMTABICLV2"
    ag_name = "SDMTabICLv2"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        X_unlabeled: pd.DataFrame | None = None,
        time_limit: float | None = None,
        sample_weight: pd.Series | None = None,
        sample_weight_val: pd.Series | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        verbosity: int = 2,
        **kwargs: Any,
    ) -> None:
        self._device = _resolve_device(num_gpus=num_gpus)
        self._feature_stypes = infer_stypes(X)
        self._target_name = str(y.name) if y.name is not None else "__target__"
        self._target_stype = (
            Stype.numerical
            if self.problem_type == "regression"
            else Stype.categorical
        )

        self.model = TabICLv2(device=self._device)
        with _autocast(self._device):
            self.model.fit(
                x=TableTensor.from_pandas(
                    df=X,
                    stypes=self._feature_stypes,
                    device=self._device,
                ),
                y=_table_from_series(
                    y,
                    name=self._target_name,
                    stype=self._target_stype,
                    device=self._device,
                ),
                num_estimators=int(self._get_model_params()["num_estimators"]),
            )

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if not hasattr(self, "model"):
            raise RuntimeError(
                "SDMTabICLv2Model must be fitted before prediction"
            )

        with _autocast(self._device):
            prediction = self.model.predict(
                TableTensor.from_pandas(
                    df=X,
                    stypes=self._feature_stypes,
                    device=self._device,
                )
            )
        values = prediction.numerical.float().cpu().numpy()
        return _prediction_to_numpy(
            values,
            problem_type=self.problem_type,
            class_labels=prediction.columns[Stype.numerical],
        )

    def _set_default_params(self) -> None:
        self._set_default_param_value("num_estimators", 8)

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        """Return the AutoGluon problem types supported by TabICLv2."""
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        return 1, 1 if torch.cuda.is_available() else 0

    def cleanup(self) -> None:
        """Release the local model cache and GPU allocations after one task."""
        if hasattr(self, "model"):
            self.model.clear()
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class FeatureSchema:
    """The SDM-owned feature contract fitted from one training frame."""

    columns: tuple[str, ...]
    stypes: dict[str, Stype]
    id_columns: tuple[str, ...]


class SDMTabICLv2System(ExternalSystemModel):
    """Run local TabICLv2 while SDM owns feature and target preprocessing."""

    def __init__(self, *, num_estimators: int = 8, **kwargs: Any) -> None:
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
        )
        if problem_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(
                f"Unsupported TabArena problem type '{problem_type}'"
            )

        self._schema = _fit_feature_schema(X)
        X = _align_features(X, schema=self._schema)
        self._device = _resolve_device(num_gpus=num_gpus)
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=self._device).manual_seed(
                random_state
            )
        self._problem_type = problem_type
        self._target_name = target_name or "__target__"
        self._target_stype = (
            Stype.numerical
            if problem_type == "regression"
            else Stype.categorical
        )
        if problem_type != "regression":
            self._class_labels_by_key = _class_labels_by_key(y)
            self._tabarena_class_order = _tabarena_class_order(
                y,
                problem_type=problem_type,
            )

        self.model = TabICLv2(device=self._device)
        with _autocast(self._device):
            self.model.fit(
                x=TableTensor.from_pandas(
                    df=X,
                    stypes=self._schema.stypes,
                    device=self._device,
                ),
                y=_table_from_series(
                    y,
                    name=self._target_name,
                    stype=self._target_stype,
                    device=self._device,
                ),
                num_estimators=self.num_estimators,
                generator=generator,
            )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        """Return indexed point predictions for a regression task."""
        if self._problem_type != "regression":
            raise RuntimeError("Classification tasks require '_predict_proba'")
        values = self._prediction_values(X)
        return pd.Series(
            values.mean(axis=-1),
            index=X.index,
            name=self._target_name,
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return probabilities in TabArena's expected raw-label order."""
        if self._problem_type == "regression":
            raise RuntimeError("Regression tasks require '_predict'")
        prediction = self._predict_table(X)
        values = prediction.numerical.float().cpu().numpy()
        labels = _labels_from_prediction_columns(
            prediction.columns[Stype.numerical],
            labels_by_key=self._class_labels_by_key,
        )
        probabilities = pd.DataFrame(
            values,
            index=X.index,
            columns=np.asarray(labels, dtype=object),
        )
        return _order_probabilities_for_tabarena(
            probabilities,
            class_order=self._tabarena_class_order,
        )

    def _prediction_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return numerical TabICLv2 output values for a query frame."""
        return self._predict_table(X).numerical.float().cpu().numpy()

    def _predict_table(self, X: pd.DataFrame) -> TableTensor:
        """Run cached TabICLv2 inference after schema validation."""
        if not hasattr(self, "model"):
            raise RuntimeError(
                "SDMTabICLv2System must be fitted before prediction"
            )
        X = _align_features(X, schema=self._schema)
        with _autocast(self._device):
            return self.model.predict(
                TableTensor.from_pandas(
                    df=X,
                    stypes=self._schema.stypes,
                    device=self._device,
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
    """Infer and validate the SDM-owned feature contract."""
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
    frame: pd.DataFrame,
    *,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """Validate a query schema and return the fitted non-ID column order."""
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
    """Map SDM's string output identifiers back to pandas labels."""
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


def _tabarena_class_order(
    y: pd.Series,
    *,
    problem_type: str,
) -> tuple[object, ...]:
    """Return the raw class order used by TabArena's evaluator."""
    label_cleaner = LabelCleaner.construct(
        problem_type=problem_type,
        y=y,
    )
    class_order = label_cleaner.ordered_class_labels
    if class_order is None:
        raise RuntimeError(
            "TabArena did not provide an ordered class-label contract"
        )
    return tuple(class_order)


def _order_probabilities_for_tabarena(
    probabilities: pd.DataFrame,
    *,
    class_order: tuple[object, ...],
) -> pd.DataFrame:
    """Validate and align probability columns for TabArena scoring."""
    expected = pd.Index(class_order)
    actual = probabilities.columns
    if actual.has_duplicates:
        raise RuntimeError(
            "TabICLv2 returned duplicate class-probability columns"
        )
    if expected.has_duplicates:
        raise RuntimeError(
            "TabArena's class-label contract contains duplicate labels"
        )

    missing = expected.difference(actual).tolist()
    unexpected = actual.difference(expected).tolist()
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing labels: {missing}")
        if unexpected:
            details.append(f"unexpected labels: {unexpected}")
        raise RuntimeError(
            "TabICLv2 probability columns do not match TabArena's class "
            "labels (" + "; ".join(details) + ")"
        )
    return probabilities.loc[:, expected]


def _prediction_to_numpy(
    values: np.ndarray,
    *,
    problem_type: str,
    class_labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    if problem_type == "regression":
        return values.mean(axis=-1)
    if class_labels is None:
        raise ValueError("Classification predictions require class labels")
    class_indices = np.asarray([int(label) for label in class_labels])
    values = values[:, np.argsort(class_indices)]
    if problem_type == "binary":
        return values[:, 1]
    return values


def _resolve_device(*, num_gpus: int | None) -> torch.device:
    if num_gpus is None or num_gpus <= 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("TabArena requested a GPU but CUDA is unavailable")
    return torch.device("cuda")


def _autocast(device: torch.device) -> torch.amp.autocast:
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def _table_from_series(
    series: pd.Series,
    *,
    name: str,
    stype: Stype,
    device: torch.device,
) -> TableTensor:
    return TableTensor.from_pandas(
        df=series.rename(name).to_frame(),
        stypes={name: stype},
        device=device,
    )


def main() -> None:
    """Run selected TabArena jobs locally and write their results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument(
        "--mode",
        choices=("autogluon-compatible", "sdm-native"),
        default="autogluon-compatible",
    )
    parser.add_argument("--outer", action="store_true")
    parser.add_argument("--subset", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()

    if args.num_estimators < 1:
        raise ValueError("'--num-estimators' must be positive")
    if args.num_cpus is not None and args.num_cpus < 1:
        raise ValueError("'--num-cpus' must be positive")
    if args.num_gpus is not None and args.num_gpus < 0:
        raise ValueError("'--num-gpus' cannot be negative")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output root {output_root} is non-empty. Choose a fresh path."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "autogluon-compatible":
        generator = ConfigGenerator(
            search_space={},
            model_cls=SDMTabICLv2Model,
            manual_configs=[{"num_estimators": args.num_estimators}],
        )
        experiments = TabArenaV0pt1ExperimentBundle(
            models=[(generator, 0)],
            outer_experiments=args.outer,
        ).build_experiments(
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
        )
    else:
        generator = SystemConfigGenerator(
            model_cls=SDMTabICLv2System,
            name="SDMTabICLv2System",
            manual_configs=[{"num_estimators": args.num_estimators}],
        )
        experiments = TabArenaV0pt1ExperimentBundle(
            models=[(generator, 0)],
            system_experiments=True,
        ).build_experiments(
            num_cpus=args.num_cpus,
            num_gpus=args.num_gpus,
        )

    context = TabArenaContext()
    jobs = context.build_jobs(
        experiments,
        task_subset=TaskSubset(
            subset=args.subset,
            dataset_names=args.datasets,
        ),
    )
    context.run_jobs(
        jobs,
        expname=output_root,
        new_result_prefix="[SDM] ",
        debug_mode=True,
    )
    results = context._registered_new_results()
    if results is None:
        raise RuntimeError("No TabArena jobs completed successfully")

    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = results.copy()
    report.insert(0, "integration_mode", args.mode)
    report.to_csv(report_dir / "results_per_split.csv", index=False)


if __name__ == "__main__":
    main()
