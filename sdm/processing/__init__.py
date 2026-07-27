"""Processors for structured data tables."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.common import (
    Identity,
    Sequential,
    StypeDispatch,
    TaskDispatch,
    Choice,
    ToNumerical,
    ShuffleColumns,
)
from sdm.processing.numerical import (
    Clip,
    ClipQuantiles,
    ClipSigma,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    Standardize,
    DropConstantColumns,
)
from sdm.processing.categorical import (
    AlignCategories,
    ShuffleCategories,
    ImputeMode,
    SortCategories,
)
from sdm.processing.datetime import AddCalendarFields
from sdm.processing.output import ReduceEstimators, Softmax
from sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
    "Identity",
    "Sequential",
    "StypeDispatch",
    "TaskDispatch",
    "Choice",
    "ToNumerical",
    "ShuffleColumns",
    "Clip",
    "ClipQuantiles",
    "ClipSigma",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "Standardize",
    "DropConstantColumns",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeMode",
    "SortCategories",
    "AddCalendarFields",
    "ReduceEstimators",
    "Softmax",
    "Recipe",
]
