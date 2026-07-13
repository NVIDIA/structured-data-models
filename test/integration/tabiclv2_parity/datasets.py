from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetCase:
    """A small deterministic train/test parity case."""

    name: str
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    target: np.ndarray
    task: str


def classification_cases() -> tuple[DatasetCase, ...]:
    """Cover the classification edge cases required by the parity audit."""
    return (
        DatasetCase(
            "binary_one_feature",
            pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]}),
            pd.DataFrame({"x": [-0.5, 0.5]}),
            np.array([10, 10, 30, 30]),
            "classification",
        ),
        DatasetCase(
            "three_string_classes_mixed_unknown_missing",
            pd.DataFrame(
                {
                    "number": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
                    "category": ["b", "a", "b", np.nan, "c", "a"],
                }
            ),
            pd.DataFrame(
                {
                    "number": [7.0, 8.0],
                    "category": ["unseen", np.nan],
                }
            ),
            np.array(["z", "a", "m", "z", "a", "m"]),
            "classification",
        ),
        DatasetCase(
            "constant_outlier_more_features_than_members",
            pd.DataFrame(
                {
                    "constant": [1.0] * 6,
                    "ordinary": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    "outlier": [0.0, 0.0, 0.0, 0.0, 0.0, 1_000.0],
                    "f3": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                    "f4": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                }
            ),
            pd.DataFrame(
                {
                    "constant": [1.0],
                    "ordinary": [6.0],
                    "outlier": [-1_000.0],
                    "f3": [-1.0],
                    "f4": [0.0],
                }
            ),
            np.array([0, 1, 2, 0, 1, 2]),
            "classification",
        ),
    )


def regression_cases() -> tuple[DatasetCase, ...]:
    """Cover the regression edge cases required by the parity audit."""
    x_train = pd.DataFrame(
        {
            "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "constant": [2.0] * 6,
        }
    )
    x_test = pd.DataFrame({"x": [-4.0, 4.0], "constant": [2.0, 2.0]})
    return (
        DatasetCase(
            "negative_and_positive",
            x_train,
            x_test,
            np.array([-9.0, -4.0, -1.0, 1.0, 4.0, 9.0]),
            "regression",
        ),
        DatasetCase(
            "constant_target",
            x_train,
            x_test,
            np.full(6, 3.5),
            "regression",
        ),
        DatasetCase(
            "near_constant_target",
            x_train,
            x_test,
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0 + 1e-7]),
            "regression",
        ),
        DatasetCase(
            "strongly_skewed_target",
            x_train,
            x_test,
            np.array([0.0, 1.0, 2.0, 4.0, 16.0, 256.0]),
            "regression",
        ),
    )
