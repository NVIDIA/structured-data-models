"""Regression-only AutoGluon adapter for the SDM TabICLv2 smoke benchmark.

This module deliberately lives under :mod:`examples` rather than an SDM public
integration package.  It keeps TabArena and AutoGluon optional while giving
subprocess and Ray workers a stable import path.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)
from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2
from sdm.models.tabiclv2.output import decode_regression_quantiles
from sdm.processing import InvertibleMixin

if TYPE_CHECKING:
    import pandas as pd
    from sdm.processing import Recipe


_DEFAULT_SEED = 0
_MAX_SEED = 2**63 - 1
_NUM_ESTIMATORS = 1


@dataclass(frozen=True)
class DeviceAllocation:
    """The one-device allocation supported by the initial smoke benchmark."""

    device: torch.device

    @classmethod
    def from_num_gpus(cls, num_gpus: int | float) -> DeviceAllocation:
        """Resolve the requested TabArena GPU count to an SDM device.

        A benchmark worker sees its allocated GPU through
        ``CUDA_VISIBLE_DEVICES``; selecting ``"cuda:0"`` therefore uses the
        worker-local allocation rather than a globally hard-coded device
        index.
        """
        if isinstance(num_gpus, bool) or num_gpus not in (0, 1):
            if isinstance(num_gpus, (int, float)) and num_gpus > 1:
                raise NotImplementedError(
                    "Multi-GPU TabICLv2 execution is not yet implemented; "
                    "request zero or one GPU."
                )
            raise ValueError(
                f"Expected 'num_gpus' to be exactly 0 or 1 (got {num_gpus!r})."
            )

        if num_gpus == 0:
            return cls(device=torch.device("cpu"))

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Fit requested one GPU, but CUDA is not available. "
                "Request num_gpus=0 to run on CPU."
            )
        return cls(device=torch.device("cuda:0"))


@contextmanager
def _fork_seed(seed: int, device: torch.device) -> Iterator[None]:
    """Seed fit-time randomness without perturbing caller RNG state."""
    devices: list[int] = []
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        devices.append(index)

    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


class SDMTabICLv2Model(AbstractTorchModel):
    """Run one locally verified SDM TabICLv2 regression estimator.

    TabArena supplies raw pandas data.  This wrapper validates it, converts it
    directly to SDM :class:`~sdm.TableTensor` values, and applies SDM's recipe.
    It intentionally never calls AutoGluon's ``self.preprocess(X)``.
    """

    ag_key = "SDM-TABICLv2"
    ag_name = "SDM-TabICLv2"
    ag_priority = 0
    preprocess_data = False

    _sdm_model: TabICLv2 | None
    _recipe: Recipe | None
    _allocation: DeviceAllocation | None
    _feature_columns: tuple[object, ...] | None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sdm_model = None
        self._recipe = None
        self._allocation = None
        self._feature_columns = None

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        """Advertise the phase-one regression-only contract."""
        return ["regression"]

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ) -> None:
        """Fit SDM preprocessing and cache the in-context training rows."""
        del num_cpus, kwargs
        self._require_regression()
        self._reset_fitted_state()

        allocation = DeviceAllocation.from_num_gpus(num_gpus)
        features, columns = self._to_feature_table(X, device=allocation.device)
        target = self._to_target_table(y, device=allocation.device)

        checkpoint_path, checkpoint_sha256, seed, num_estimators = (
            self._model_config()
        )
        with _fork_seed(seed, allocation.device):
            model = TabICLv2(pretrained=False, device=allocation.device)
            model.load_regression_checkpoint(
                checkpoint_path,
                checkpoint_sha256,
            )
            recipe = model.default_recipe()
            model_features = recipe.features.fit_transform(features)
            model_target = recipe.target.fit_transform(target)
            model.fit(
                x=model_features,
                y=model_target,
                num_estimators=num_estimators,
            )

        self._sdm_model = model
        # ``AbstractTorchModel`` and some AutoGluon tooling expect this name.
        self.model = model
        self._recipe = recipe
        self._allocation = allocation
        self._feature_columns = columns

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Return one regression scalar per row through AutoGluon's API."""
        del kwargs
        model, recipe, allocation, _ = self._require_fitted()
        features, _ = self._to_feature_table(
            X,
            device=allocation.device,
            expected_columns=self._feature_columns,
        )
        model_features = recipe.features.transform(features)
        quantiles = model.predict(model_features)
        target_transform = recipe.target
        if not isinstance(target_transform, InvertibleMixin):
            raise RuntimeError(
                "The TabICLv2 regression target recipe must support inverse "
                "transformation."
            )
        predictions = decode_regression_quantiles(quantiles, target_transform)

        if predictions.ndim != 1 or predictions.size(0) != len(X):
            raise RuntimeError(
                "SDM TabICLv2 regression decoder returned an invalid "
                f"prediction shape {tuple(predictions.shape)} "
                f"for {len(X)} rows."
            )
        if not torch.isfinite(predictions).all():
            raise RuntimeError("SDM TabICLv2 produced non-finite predictions.")

        return (
            predictions.detach().to(device="cpu", dtype=torch.float32).numpy()
        )

    def _get_default_resources(self) -> tuple[int, int]:
        """Use the worker's physical CPUs and at most one CUDA device."""
        return (
            ResourceManager.get_cpu_count(only_physical_cores=True),
            min(1, ResourceManager.get_gpu_count_torch(cuda_only=True)),
        )

    def get_minimum_resources(
        self,
        is_gpu_available: bool = False,
    ) -> dict[str, int]:
        """CPU is sufficient; a caller may explicitly allocate one GPU."""
        del is_gpu_available
        return {"num_cpus": 1, "num_gpus": 0}

    def get_device(self) -> str:
        """Return the device selected during fitting."""
        _, _, allocation, _ = self._require_fitted()
        return allocation.device.type

    def _set_device(self, device: str) -> None:
        """Move SDM model, recipes, and cache for AutoGluon save/load hooks."""
        model, recipe, _, _ = self._require_fitted()
        device_obj = torch.device(device)
        model.to(device_obj)
        if model._caches is not None:
            model._caches = [cache.to(device_obj) for cache in model._caches]
        for processor in (recipe.features, recipe.target, recipe.output):
            processor.to(device_obj)
        self._allocation = DeviceAllocation(device=device_obj)

    # SDM conversion and schema guards ######################################

    def _to_feature_table(
        self,
        X: pd.DataFrame,
        *,
        device: torch.device,
        expected_columns: tuple[object, ...] | None = None,
    ) -> tuple[TableTensor, tuple[object, ...]]:
        self._require_dataframe(X, name="X")
        if not X.columns.is_unique:
            raise ValueError(
                "SDM TabICLv2 requires unique feature column names."
            )
        if len(X.columns) == 0:
            raise ValueError(
                "SDM TabICLv2 requires at least one feature column."
            )

        columns = tuple(X.columns.tolist())
        if expected_columns is not None and columns != expected_columns:
            self._raise_schema_mismatch(expected_columns, columns)

        stypes = infer_stypes(X)
        unsupported = {
            str(column): stype.value
            for column, stype in stypes.items()
            if stype is not Stype.numerical
        }
        if unsupported:
            formatted = ", ".join(
                f"{name} ({stype})" for name, stype in unsupported.items()
            )
            raise ValueError(
                "SDM TabICLv2 smoke currently accepts only numerical "
                f"features; found {formatted}."
            )

        return (
            TableTensor.from_pandas(df=X, stypes=stypes, device=device),
            columns,
        )

    @staticmethod
    def _to_target_table(
        y: pd.Series,
        *,
        device: torch.device,
    ) -> TableTensor:
        import pandas as pd

        if not isinstance(y, pd.Series):
            raise TypeError(
                "Expected regression target 'y' to be a pandas Series "
                f"(got {type(y).__name__})."
            )
        if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(
            y
        ):
            raise ValueError(
                "SDM TabICLv2 requires a numerical regression target."
            )
        if y.isna().any():
            raise ValueError(
                "SDM TabICLv2 does not accept missing regression targets."
            )

        target_name = "__sdm_target__"
        target = y.to_frame(name=target_name)
        return TableTensor.from_pandas(
            df=target,
            stypes={target_name: Stype.numerical},
            device=device,
        )

    @staticmethod
    def _require_dataframe(X: object, *, name: str) -> None:
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Expected '{name}' to be a pandas DataFrame "
                f"(got {type(X).__name__})."
            )

    @staticmethod
    def _raise_schema_mismatch(
        expected: tuple[object, ...],
        actual: tuple[object, ...],
    ) -> None:
        expected_set = set(expected)
        actual_set = set(actual)
        missing = [column for column in expected if column not in actual_set]
        unexpected = [
            column for column in actual if column not in expected_set
        ]
        if missing or unexpected:
            raise ValueError(
                "Feature schema differs from fit: "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )
        raise ValueError(
            "Feature column order differs from fit. SDM TabICLv2 requires "
            "the original numerical feature order."
        )

    # Adapter state ##########################################################

    def _require_regression(self) -> None:
        if self.problem_type != "regression":
            raise ValueError(
                "SDMTabICLv2Model supports regression only "
                f"(got problem_type={self.problem_type!r})."
            )

    def _model_config(self) -> tuple[Path, str, int, int]:
        params = self._get_model_params()
        checkpoint_path = params.get("checkpoint_path")
        checkpoint_sha256 = params.get("checkpoint_sha256")
        seed = params.get("seed", _DEFAULT_SEED)
        num_estimators = params.get("num_estimators", _NUM_ESTIMATORS)

        if not isinstance(checkpoint_path, (str, Path)):
            raise ValueError(
                "Set the required 'checkpoint_path' model hyperparameter to "
                "a local TabICLv2 regression checkpoint."
            )
        if not isinstance(checkpoint_sha256, str):
            raise ValueError(
                "Set the required 'checkpoint_sha256' model hyperparameter."
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= _MAX_SEED
        ):
            raise ValueError(
                "Expected 'seed' to be an integer between 0 and "
                f"{_MAX_SEED} (got {seed!r})."
            )
        if (
            isinstance(num_estimators, bool)
            or not isinstance(num_estimators, int)
            or num_estimators != _NUM_ESTIMATORS
        ):
            raise ValueError(
                "The phase-one SDM TabICLv2 smoke requires exactly one "
                f"estimator (got {num_estimators!r})."
            )
        return (
            Path(checkpoint_path),
            checkpoint_sha256,
            seed,
            num_estimators,
        )

    def _require_fitted(
        self,
    ) -> tuple[TabICLv2, Recipe, DeviceAllocation, tuple[object, ...]]:
        if (
            self._sdm_model is None
            or self._recipe is None
            or self._allocation is None
            or self._feature_columns is None
        ):
            raise RuntimeError(
                "SDMTabICLv2Model is not fitted; call fit before predicting."
            )
        return (
            self._sdm_model,
            self._recipe,
            self._allocation,
            self._feature_columns,
        )

    def _reset_fitted_state(self) -> None:
        if self._sdm_model is not None:
            self._sdm_model.clear()
        self._sdm_model = None
        self._recipe = None
        self._allocation = None
        self._feature_columns = None
        self.model = None


__all__ = ["DeviceAllocation", "SDMTabICLv2Model"]
