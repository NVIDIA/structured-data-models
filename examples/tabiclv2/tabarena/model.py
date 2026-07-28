"""TabICLv2 model adapter for TabArena."""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
import torch
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.metrics import Scorer
from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2
from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.task.metadata import ValidationMetadata


class SDMTabICLv2System(ExternalSystemModel):
    """Expose TabICLv2 through TabArena's external-system interface."""

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
