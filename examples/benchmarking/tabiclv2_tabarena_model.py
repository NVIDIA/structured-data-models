"""Regression and binary-classification AutoGluon adapter for SDM TabICLv2.

This module deliberately lives under :mod:`examples` rather than an SDM public
integration package.  It keeps TabArena and AutoGluon optional while giving
subprocess and Ray workers a stable import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)
from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2
from sdm.models.tabiclv2.output import reduce_regression_quantiles
from sdm.processing import Recipe

if TYPE_CHECKING:
    import pandas as pd


_DEFAULT_SEED = 0
_MAX_SEED = 2**63 - 1
_NUM_ESTIMATORS = 1
DEFAULT_ATTENTION_BATCH_SIZE_LIMIT = 128
DEFAULT_PREDICTION_BATCH_SIZE = 512


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


class SDMTabICLv2Model(AbstractTorchModel):
    """Run one locally verified SDM TabICLv2 regression or classifier.

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
    _class_labels: tuple[str, ...] | None
    _prediction_batch_size: int | None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sdm_model = None
        self._recipe = None
        self._allocation = None
        self._feature_columns = None
        self._class_labels = None
        self._prediction_batch_size = None

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        """Advertise the supported TabArena problem types."""
        return ["binary", "multiclass", "regression"]

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
        problem_type = self._require_supported_problem_type()
        self._reset_fitted_state()

        allocation = DeviceAllocation.from_num_gpus(num_gpus)
        features, columns = self._to_feature_table(X, device=allocation.device)
        target, class_labels = self._to_target_table(
            y,
            device=allocation.device,
            problem_type=problem_type,
        )

        (
            checkpoint_path,
            checkpoint_sha256,
            seed,
            num_estimators,
            batch_size_limit,
            prediction_batch_size,
        ) = self._model_config()
        generator = torch.Generator(device=allocation.device)
        generator.manual_seed(seed)
        model = TabICLv2(
            pretrained=False,
            device=allocation.device,
            batch_size_limit=batch_size_limit,
        )
        if problem_type == "regression":
            model.load_regression_checkpoint(
                checkpoint_path,
                checkpoint_sha256,
            )
        else:
            model.load_classifier_checkpoint(
                checkpoint_path,
                checkpoint_sha256,
            )
        recipe = self._build_recipe(model)
        model.fit(
            x=features,
            y=target,
            recipe=recipe,
            num_estimators=num_estimators,
            generator=generator,
        )
        if model._caches is None:
            raise RuntimeError("SDM TabICLv2 did not cache the fitted recipe.")
        fitted_recipe = cast(Recipe, model._caches[0]["recipe"])

        self._sdm_model = model
        # ``AbstractTorchModel`` and some AutoGluon tooling expect this name.
        self.model = model
        self._recipe = fitted_recipe
        self._allocation = allocation
        self._feature_columns = columns
        self._class_labels = class_labels
        self._prediction_batch_size = prediction_batch_size

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Return regression predictions or class probabilities for ``X``."""
        del kwargs
        model, _, allocation, _, prediction_batch_size = self._require_fitted()
        features, _ = self._to_feature_table(
            X,
            device=allocation.device,
            expected_columns=self._feature_columns,
        )
        output = self._predict_in_chunks(
            model,
            features,
            prediction_batch_size=prediction_batch_size,
        )
        if self.problem_type == "regression":
            member_predictions = reduce_regression_quantiles(output.numerical)
            if member_predictions.ndim != 2:
                raise RuntimeError(
                    "SDM TabICLv2 returned an invalid regression prediction "
                    f"shape {tuple(member_predictions.shape)}."
                )
            predictions = member_predictions.mean(dim=0)
            if predictions.ndim != 1 or predictions.size(0) != len(X):
                raise RuntimeError(
                    "SDM TabICLv2 regression decoder returned an invalid "
                    f"prediction shape {tuple(predictions.shape)} "
                    f"for {len(X)} rows."
                )
            if not torch.isfinite(predictions).all():
                raise RuntimeError(
                    "SDM TabICLv2 produced non-finite predictions."
                )
            return (
                predictions.detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .numpy()
            )

        class_labels = self._class_labels
        if class_labels is None:
            raise RuntimeError(
                "Classification adapter is missing its fitted class labels."
            )
        probabilities = output.numerical
        if probabilities.ndim != 3 or probabilities.size(1) != len(X):
            raise RuntimeError(
                "SDM TabICLv2 classifier returned an invalid prediction "
                f"shape {tuple(probabilities.shape)} for {len(X)} rows."
            )
        column_positions = {
            column: index
            for index, column in enumerate(output.columns[Stype.numerical])
        }
        try:
            positions = [column_positions[label] for label in class_labels]
        except KeyError as error:
            raise RuntimeError(
                "SDM TabICLv2 classifier output columns do not match the "
                f"fitted classes {class_labels!r}."
            ) from error
        if len(positions) != len(class_labels) or probabilities.size(
            -1
        ) != len(class_labels):
            raise RuntimeError(
                "SDM TabICLv2 classifier output width does not match its "
                f"{len(class_labels)} fitted classes."
            )
        probabilities = probabilities[..., positions].mean(dim=0)
        totals = probabilities.sum(dim=-1, keepdim=True)
        if (
            not torch.isfinite(probabilities).all()
            or not torch.isfinite(totals).all()
            or (totals <= 0).any()
        ):
            raise RuntimeError(
                "SDM TabICLv2 produced invalid classification probabilities."
            )
        probabilities = probabilities / totals
        return (
            probabilities.detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .numpy()
        )

    def _get_default_resources(self) -> tuple[int, int]:
        """Use the worker's physical CPUs and at most one CUDA device."""
        return (
            ResourceManager.get_cpu_count(only_physical_cores=True),
            min(1, ResourceManager.get_gpu_count_torch(cuda_only=True)),
        )

    def _build_recipe(self, model: TabICLv2) -> Recipe:
        """Build the preprocessing recipe used for this fit.

        The default adapter continues to use SDM's native recipe. Benchmark
        subclasses may override this protected hook to select a fully explicit
        comparison recipe without changing the normal smoke configuration.
        """
        return model.default_recipe()

    def get_minimum_resources(
        self,
        is_gpu_available: bool = False,
    ) -> dict[str, int]:
        """CPU is sufficient; a caller may explicitly allocate one GPU."""
        del is_gpu_available
        return {"num_cpus": 1, "num_gpus": 0}

    def get_device(self) -> str:
        """Return the device selected during fitting."""
        _, _, allocation, _, _ = self._require_fitted()
        return allocation.device.type

    def _set_device(self, device: str) -> None:
        """Move SDM model, recipes, and cache for AutoGluon save/load hooks."""
        model, _, _, _, _ = self._require_fitted()
        device_obj = torch.device(device)
        model.to(device_obj)
        if model._caches is not None:
            model._caches = [cache.to(device_obj) for cache in model._caches]
            for cache in model._caches:
                cached_recipe = cast(Recipe, cache["recipe"])
                for processor in (
                    cached_recipe.features,
                    cached_recipe.target,
                    cached_recipe.output,
                ):
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
            if stype not in {Stype.numerical, Stype.categorical}
        }
        if unsupported:
            formatted = ", ".join(
                f"{name} ({stype})" for name, stype in unsupported.items()
            )
            raise ValueError(
                "SDM TabICLv2 supports only numerical and categorical "
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
        problem_type: Literal["binary", "multiclass", "regression"],
    ) -> tuple[TableTensor, tuple[str, ...] | None]:
        import pandas as pd

        if not isinstance(y, pd.Series):
            raise TypeError(
                "Expected target 'y' to be a pandas Series "
                f"(got {type(y).__name__})."
            )
        if y.isna().any():
            raise ValueError("SDM TabICLv2 does not accept missing targets.")

        target_name = "__sdm_target__"
        target = y.to_frame(name=target_name)
        if problem_type == "regression":
            if pd.api.types.is_bool_dtype(
                y
            ) or not pd.api.types.is_numeric_dtype(y):
                raise ValueError(
                    "SDM TabICLv2 requires a numerical regression target."
                )
            return (
                TableTensor.from_pandas(
                    df=target,
                    stypes={target_name: Stype.numerical},
                    device=device,
                ),
                None,
            )

        class_values = np.unique(y.to_numpy())
        expected_class_range = (
            class_values.size == 2
            if problem_type == "binary"
            else 3 <= class_values.size <= 10
        )
        if not expected_class_range:
            expected = "exactly two" if problem_type == "binary" else "3 to 10"
            raise ValueError(
                f"SDM TabICLv2 {problem_type} classification requires "
                f"{expected} training classes (got {class_values.size})."
            )
        return (
            TableTensor.from_pandas(
                df=target,
                stypes={target_name: Stype.categorical},
                device=device,
            ),
            tuple(str(value) for value in class_values.tolist()),
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
            "the original feature order."
        )

    # Adapter state ##########################################################

    def _require_supported_problem_type(
        self,
    ) -> Literal["binary", "multiclass", "regression"]:
        if self.problem_type not in {"binary", "multiclass", "regression"}:
            raise ValueError(
                "SDMTabICLv2Model supports binary classification, multiclass "
                f"classification, and regression (got "
                f"problem_type={self.problem_type!r})."
            )
        return cast(
            Literal["binary", "multiclass", "regression"],
            self.problem_type,
        )

    @staticmethod
    def _predict_in_chunks(
        model: TabICLv2,
        features: TableTensor,
        *,
        prediction_batch_size: int,
    ) -> TableTensor:
        """Predict ordered test-row chunks against one fitted cache."""
        if len(features) <= prediction_batch_size:
            return model.predict(features)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(features), prediction_batch_size):
            stop = min(start + prediction_batch_size, len(features))
            outputs.append(model.predict(features[start:stop]))
        return cast(TableTensor, torch.cat(outputs, dim=1))

    def _model_config(self) -> tuple[Path, str, int, int, int, int]:
        params = self._get_model_params()
        checkpoint_path = params.get("checkpoint_path")
        checkpoint_sha256 = params.get("checkpoint_sha256")
        seed = params.get("seed", _DEFAULT_SEED)
        num_estimators = params.get("num_estimators", _NUM_ESTIMATORS)
        batch_size_limit = params.get(
            "attention_batch_size_limit",
            DEFAULT_ATTENTION_BATCH_SIZE_LIMIT,
        )

        prediction_batch_size = params.get(
            "prediction_batch_size",
            DEFAULT_PREDICTION_BATCH_SIZE,
        )

        if not isinstance(checkpoint_path, (str, Path)):
            raise ValueError(
                "Set the required 'checkpoint_path' model hyperparameter to "
                "a local TabICLv2 checkpoint."
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
        if (
            isinstance(batch_size_limit, bool)
            or not isinstance(batch_size_limit, int)
            or batch_size_limit < 1
        ):
            raise ValueError(
                "Expected attention_batch_size_limit to be a positive "
                f"integer (got {batch_size_limit!r})."
            )
        if (
            isinstance(prediction_batch_size, bool)
            or not isinstance(prediction_batch_size, int)
            or prediction_batch_size < 1
        ):
            raise ValueError(
                "Expected prediction_batch_size to be a positive integer "
                f"(got {prediction_batch_size!r})."
            )

        return (
            Path(checkpoint_path),
            checkpoint_sha256,
            seed,
            num_estimators,
            batch_size_limit,
            prediction_batch_size,
        )

    def _require_fitted(
        self,
    ) -> tuple[TabICLv2, Recipe, DeviceAllocation, tuple[object, ...], int]:
        if (
            self._sdm_model is None
            or self._recipe is None
            or self._allocation is None
            or self._feature_columns is None
            or self._prediction_batch_size is None
        ):
            raise RuntimeError(
                "SDMTabICLv2Model is not fitted; call fit before predicting."
            )
        return (
            self._sdm_model,
            self._recipe,
            self._allocation,
            self._feature_columns,
            self._prediction_batch_size,
        )

    def _reset_fitted_state(self) -> None:
        if self._sdm_model is not None:
            self._sdm_model.clear()
        self._sdm_model = None
        self._recipe = None
        self._allocation = None
        self._feature_columns = None
        self._class_labels = None
        self._prediction_batch_size = None
        self.model = None


__all__ = ["DeviceAllocation", "SDMTabICLv2Model"]
