r"""Run TabICLv2 on TabArena.

$ uv run --group example-tabarena python examples/tabiclv2/tab_arena.py

The output directory must be empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

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
from tabarena.contexts import TabArenaContext
from tabarena.end_to_end import EndToEnd
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
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
        **_: object,
    ) -> Self:
        random_state = 42 if random_state is None else random_state
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        generator = torch.Generator(device=self._device).manual_seed(
            random_state
        )

        self.stypes = infer_stypes(X)
        target_name = target_name or "__target__"
        if problem_type == "regression":
            target_stype = Stype.numerical
        else:
            target_stype = Stype.categorical
            if y.isna().any():
                raise ValueError(
                    "Classification targets must not contain missing values"
                )
            class_labels_by_key = {}
            for label in pd.unique(y):
                key = str(label)
                if key in class_labels_by_key:
                    raise ValueError(
                        "Classification labels have ambiguous string "
                        "representations: "
                        f"{class_labels_by_key[key]!r} and {label!r} "
                        f"both map to {key!r}"
                    )
                class_labels_by_key[key] = label

            label_cleaner = LabelCleaner.construct(
                problem_type=problem_type,
                y=y,
            )
            self._class_labels_by_key = {
                str(label): label
                for label in label_cleaner.ordered_class_labels
            }

        self.model = TabICLv2(device=self._device)

        table_x = TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        table_y = TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: target_stype},
            device=self._device,
        )
        self.model.fit(
            x=table_x,
            y=table_y,
            num_estimators=8,
            generator=generator,
        )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        table_x = TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        out = self.model.predict(table_x)
        values = out.numerical.float().mean(dim=-1).cpu().numpy()
        return pd.Series(values, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        table_x = TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        prediction = self.model.predict(table_x)
        values = prediction.numerical.float().cpu().numpy()
        labels = [
            self._class_labels_by_key[column]
            for column in prediction.columns[Stype.numerical]
        ]
        probabilities = pd.DataFrame(
            values,
            index=X.index,
            columns=np.asarray(labels, dtype=object),
        )
        return probabilities.loc[
            :,
            pd.Index(tuple(self._class_labels_by_key.values())),
        ]


def main() -> None:
    output_root = Path(__file__).parent / "tabarena_out" / "TabICLv2"
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
    ).build_experiments()

    context = TabArenaContext()
    jobs = context.build_jobs(experiments)
    raw_results = context.run_jobs(
        jobs,
        expname=output_root,
        register=False,
        debug_mode=True,
    )
    results = EndToEnd.from_raw_to_results_df(
        results_lst=raw_results,
        task_metadata=context.task_metadata_collection,
        new_result_prefix="[SDM] ",
    )

    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_dir / "results_per_split.csv", index=False)


if __name__ == "__main__":
    main()
