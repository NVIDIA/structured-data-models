"""Cross-stype transforms that operate on multiple column types."""

from sdm.processing.common.identity import Identity
from sdm.processing.common.sequential import Sequential
from sdm.processing.common.choice import Choice
from sdm.processing.common.stype_dispatch import StypeDispatch
from sdm.processing.common.task_dispatch import TaskDispatch
from sdm.processing.common.feature_permute import FeaturePermute
from sdm.processing.common.to_numerical import ToNumerical

__all__ = [
    "Identity",
    "Sequential",
    "Choice",
    "StypeDispatch",
    "TaskDispatch",
    "FeaturePermute",
    "ToNumerical",
]
