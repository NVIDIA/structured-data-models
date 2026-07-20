"""AutoGluon model wrapper for the local SDM TabICLv2 implementation."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from autogluon.core.models import AbstractModel
from sdm import Stype, StypeLike, TableTensor, infer_stypes
from sdm.models import TabICLv2

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
        self._autocast_enabled = _autocast_enabled(
            precision=cast(str, self._get_model_params()["precision"]),
            device=self._device,
        )

        self.model = TabICLv2(device=self._device)
        with torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._autocast_enabled,
        ):
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

        with torch.amp.autocast(
            device_type=self._device.type,
            dtype=torch.bfloat16,
            enabled=self._autocast_enabled,
        ):
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
        self._set_default_param_value("precision", "auto")

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


def _autocast_enabled(
    *,
    precision: Literal["auto", "bf16", "fp32"] | str,
    device: torch.device,
) -> bool:
    """Resolve precision for TabICLv2 forward passes."""
    if precision == "fp32":
        return False
    if precision not in {"auto", "bf16"}:
        raise ValueError(
            "'precision' must be one of 'auto', 'bf16', or 'fp32' "
            f"(got {precision!r})"
        )
    if device.type != "cuda":
        if precision == "bf16":
            raise ValueError("'precision=bf16' requires a CUDA device")
        return False
    if torch.cuda.is_bf16_supported():
        return True
    if precision == "bf16":
        raise RuntimeError(
            "'precision=bf16' requires CUDA hardware with bfloat16 support"
        )
    return False


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
