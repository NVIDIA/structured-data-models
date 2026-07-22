r"""Run TabICLv2 on TabArena.

$ uv run --group example-tabarena python examples/tabiclv2_tabarena.py \
    --output-root outputs/tabiclv2-tabarena \
    --subset lite \
    --datasets blood-transfusion-service-center \
    --num-estimators 1 \
    --num-cpus 1 \
    --num-gpus 0

The output directory must be empty.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.metrics import Scorer
from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2
from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.benchmark.task.metadata.collection import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import SystemConfigGenerator


@dataclass(frozen=True)
class FeatureSchema:
    columns: tuple[str, ...]
    stypes: dict[str, Stype]
    id_columns: tuple[str, ...]


class SDMTabICLv2System(ExternalSystemModel):
    def __init__(self, *, num_estimators: int = 8, **kwargs: Any) -> None:
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
        with torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._device.type == "cuda",
        ):
            self.model.fit(
                x=TableTensor.from_pandas(
                    df=X,
                    stypes=self._schema.stypes,
                    device=self._device,
                ),
                y=TableTensor.from_pandas(
                    df=y.rename(self._target_name).to_frame(),
                    stypes={self._target_name: self._target_stype},
                    device=self._device,
                ),
                num_estimators=self.num_estimators,
                generator=generator,
            )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self._problem_type != "regression":
            raise RuntimeError("Classification tasks require '_predict_proba'")
        values = self._predict_table(X).numerical.float().cpu().numpy()
        return pd.Series(
            values.mean(axis=-1),
            index=X.index,
            name=self._target_name,
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
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

    def _predict_table(self, X: pd.DataFrame) -> TableTensor:
        if not hasattr(self, "model"):
            raise RuntimeError(
                "SDMTabICLv2System must be fitted before prediction"
            )
        X = _align_features(X, schema=self._schema)
        with torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._device.type == "cuda",
        ):
            return self.model.predict(
                TableTensor.from_pandas(
                    df=X,
                    stypes=self._schema.stypes,
                    device=self._device,
                )
            )


def _fit_feature_schema(frame: pd.DataFrame) -> FeatureSchema:
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
            f"{columns}. Add an SDM datetime recipe before running this task."
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

    return frame.loc[:, list(schema.columns)]


def _class_labels_by_key(y: pd.Series) -> dict[str, object]:
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


def _resolve_device(*, num_gpus: int | None) -> torch.device:
    if num_gpus is None or num_gpus <= 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("TabArena requested a GPU but CUDA is unavailable")
    return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-estimators", type=int, default=8)
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--subset", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()

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
    results.to_csv(report_dir / "results_per_split.csv", index=False)


if __name__ == "__main__":
    main()
