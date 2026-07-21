"""Run the repository-local TabICLv2 model on TabArena."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from examples.tabiclv2_tabarena_sdm import SDMTabICLv2System
from sdm import Stype, StypeLike, TableTensor, infer_stypes
from sdm.models import TabICLv2
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata.collection import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator, SystemConfigGenerator

if TYPE_CHECKING:
    import pandas as pd


class SDMTabICLv2Model(AbstractModel):
    """Expose local :class:`sdm.models.TabICLv2` through AutoGluon."""

    ag_key = "SDMTABICLV2"
    ag_name = "SDMTabICLv2"

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs: object,
    ) -> None:
        del kwargs
        self._device = _resolve_device(num_gpus=num_gpus)
        self._feature_stypes = infer_stypes(X)
        self._target_name = str(y.name) if y.name is not None else "__target__"
        self._target_stype = (
            Stype.numerical
            if self.problem_type == "regression"
            else Stype.categorical
        )

        self.model = TabICLv2(device=self._device)
        self.model.fit(
            x=_table_from_frame(
                X,
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

    def _predict_proba(self, X: pd.DataFrame, **kwargs: object) -> np.ndarray:
        del kwargs
        if not hasattr(self, "model"):
            raise RuntimeError(
                "SDMTabICLv2Model must be fitted before prediction"
            )

        prediction = self.model.predict(
            _table_from_frame(
                X,
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


def _resolve_device(*, num_gpus: int) -> torch.device:
    if num_gpus <= 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("TabArena requested a GPU but CUDA is unavailable")
    return torch.device("cuda")


def _table_from_frame(
    frame: pd.DataFrame,
    *,
    stypes: dict[str, StypeLike],
    device: torch.device,
) -> TableTensor:
    return TableTensor.from_pandas(
        df=frame,
        stypes=stypes,
        device=device,
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
