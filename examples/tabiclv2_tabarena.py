r"""Run TabICLv2 on TabArena.

$ uv run --group example-tabarena python examples/tabiclv2_tabarena.py \
    --output-root outputs/tabiclv2-tabarena \
    --subset lite \
    --datasets blood-transfusion-service-center \
    --num-cpus 1 \
    --num-gpus 0

The output directory must be empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


class SDMTabICLv2System(ExternalSystemModel):
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

        self._feature_stypes = infer_stypes(X)
        self._device = _resolve_device(num_gpus=num_gpus)
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=self._device).manual_seed(
                random_state
            )
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
        self.model.fit(
            x=TableTensor.from_pandas(
                df=X,
                stypes=self._feature_stypes,
                device=self._device,
            ),
            y=TableTensor.from_pandas(
                df=y.rename(self._target_name).to_frame(),
                stypes={self._target_name: self._target_stype},
                device=self._device,
            ),
            num_estimators=8,
            generator=generator,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        values = (
            self._predict_table(X).numerical.float().mean(dim=-1).cpu().numpy()
        )
        return pd.Series(
            values,
            index=X.index,
            name=self._target_name,
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
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
        return self.model.predict(
            TableTensor.from_pandas(
                df=X,
                stypes=self._feature_stypes,
                device=self._device,
            )
        )


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
        manual_configs=[{}],
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
