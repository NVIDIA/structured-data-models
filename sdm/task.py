"""Prediction task types for structured data models."""

from enum import Enum


class TaskType(str, Enum):
    r"""Prediction task type.

    Selects how a model interprets and embeds the target ``y``.

    Attributes:
        classification: Predict a discrete class label. Targets are integer
            class indices.
        regression: Predict a continuous scalar value. Targets are real
            numbers.
    """

    classification = "classification"
    regression = "regression"
