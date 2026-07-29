"""TabICLv2 model adapter for TabArena."""

from __future__ import annotations

from typing import Self

import pandas as pd
import torch
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.metrics import Scorer
from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.task.metadata import ValidationMetadata

from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2


class SDMTabICLv2System(ExternalSystemModel):
    """Expose TabICLv2 through TabArena's external-system interface.

    Args:
        kwargs: Arguments forwarded to TabArena's external-system model.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = TabICLv2(device=self._device)

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
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=self._device).manual_seed(
                random_state
            )

        self.stypes = infer_stypes(X)
        target_name = target_name or "__target__"
        if problem_type == "regression":
            target_stype = Stype.numerical
        else:
            target_stype = Stype.categorical
            label_cleaner = LabelCleaner.construct(
                problem_type=problem_type,
                y=y,
            )
            self._class_labels_by_key = {
                str(label): label
                for label in label_cleaner.ordered_class_labels
            }

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
        probabilities = self.model.predict(table_x).to_pandas()
        probabilities.index = X.index
        probabilities = probabilities.rename(
            columns=self._class_labels_by_key,
        )
        return probabilities.reindex(
            columns=tuple(self._class_labels_by_key.values()),
        )
