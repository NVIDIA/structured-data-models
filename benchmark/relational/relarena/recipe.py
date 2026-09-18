# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe variants used by the RelArena KumoRelational grid."""

from __future__ import annotations

from collections.abc import Callable

from relbench.base import TaskType

import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.models import KumoRelational
from sdm.models.callback import Callback
from sdm.processing import InvertibleMixin, Processor


class SignedLog1p(Processor, InvertibleMixin):
    """Apply ``sign(x) * log1p(abs(x))`` and its inverse."""

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        values = table.numerical
        return table.replace_blocks(
            numerical=values.sign() * values.abs().log1p()
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        values = table.numerical
        return table.replace_blocks(
            numerical=values.sign() * values.abs().expm1()
        )


class MedianQuantileOutput(Callback):
    """Select the median prediction before target inversion."""

    def on_model_forward_end(
        self,
        model: object,
        out: TableTensor,
    ) -> TableTensor:
        del model
        columns = out.columns.get(Stype.numerical, ())
        if "q500" not in columns or len(columns) == 1:
            return out
        return out[["q500"]]


def _calendar_prepend(recipe: sp.Recipe) -> sp.Recipe:
    return recipe.prepend_features(
        sp.TableDispatch(
            related=sp.StypeDispatch(
                datetime=sp.AddCalendarFields(
                    ("minute", "hour", "weekday", "day_of_month", "month"),
                ),
            ),
        )
    )


def _four_way_features() -> sp.Choice:
    return sp.Choice(
        sp.Identity(),
        sp.PowerTransform(),
        sp.QuantileTransform(output_distribution="normal"),
        sp.QuantileTransform(output_distribution="uniform"),
        method="round_robin",
    )


def _base_recipe(
    *,
    feature_transform: Processor,
    target_transform: Processor | None = None,
) -> sp.Recipe:
    if target_transform is None:
        target_transform = sp.Standardize()
    return _calendar_prepend(
        sp.Recipe(
            features=[
                sp.StypeDispatch(
                    categorical=[
                        sp.AlignCategories(sort_by="value"),
                        sp.ToNumerical(),
                    ],
                ),
                sp.StypeDispatch(
                    numerical=[
                        sp.ImputeMean(),
                        sp.DropConstantColumns(),
                        sp.Standardize(eps=1e-6),
                        sp.Clip(min_value=-100.0, max_value=100.0),
                        feature_transform,
                        sp.ClipSigma(threshold=4.0),
                        sp.ShuffleColumns(method="latin"),
                    ],
                ),
            ],
            target=[
                sp.StypeDispatch(
                    categorical=[
                        sp.AlignCategories(),
                        sp.ShuffleCategories(method="shift"),
                    ],
                    numerical=target_transform,
                ),
            ],
            output=[
                sp.ReduceEstimators(method="mean"),
                sp.TaskDispatch(
                    classification=sp.Softmax(temperature=0.9),
                ),
            ],
        )
    )


def _default() -> sp.Recipe:
    return KumoRelational.default_recipe()


def _softmax_mean() -> sp.Recipe:
    recipe = _default()
    recipe.output = sp.EnsembleProcessor.as_processor(
        [
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=0.9),
            ),
            sp.ReduceEstimators(method="mean"),
        ]
    )
    recipe._validate_output()
    return recipe


def _classification_quantile() -> sp.Recipe:
    return _base_recipe(
        feature_transform=sp.QuantileTransform(output_distribution="normal")
    )


def _classification_identity() -> sp.Recipe:
    return _base_recipe(feature_transform=sp.Identity())


def _regression_standardize() -> sp.Recipe:
    return _base_recipe(
        feature_transform=_four_way_features(),
    )


def _regression_quantile() -> sp.Recipe:
    return _base_recipe(
        feature_transform=_four_way_features(),
        target_transform=sp.QuantileTransform(
            output_distribution="normal",
        ),
    )


def _regression_robust() -> sp.Recipe:
    return _base_recipe(
        feature_transform=_four_way_features(),
        target_transform=sp.Sequential(
            SignedLog1p(),
            sp.Standardize(),
        ),
    )


_RECIPE_BUILDERS: dict[
    str,
    tuple[Callable[[], sp.Recipe], Callable[[], sp.Recipe]],
] = {
    "base": (_default, _default),
    "robust": (_softmax_mean, _regression_robust),
    "quantile": (_classification_quantile, _regression_quantile),
    "best_hard": (_default, _regression_robust),
    "cls_base_or_reg_quantile": (_default, _regression_quantile),
    "softmax_mean_or_4way_standardize": (
        _softmax_mean,
        _regression_standardize,
    ),
    "cls_identity_or_reg_base": (_classification_identity, _default),
}


def recipe_by_name(name: str, task_type: TaskType) -> sp.Recipe:
    """Build the task-specific recipe named by the grid."""
    try:
        classification, regression = _RECIPE_BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(_RECIPE_BUILDERS)
        raise ValueError(f"unknown recipe {name!r}; known: {known}") from exc
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return classification()
    if task_type == TaskType.REGRESSION:
        return regression()
    raise ValueError(f"unsupported task type: {task_type}")


def recipe_callbacks(
    name: str,
    task_type: TaskType,
) -> tuple[MedianQuantileOutput, ...]:
    """Return callbacks needed before target inverse transformation."""
    if task_type == TaskType.REGRESSION and name in {
        "robust",
        "quantile",
        "cls_base_or_reg_quantile",
    }:
        return (MedianQuantileOutput(),)
    return ()
