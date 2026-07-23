"""Fittable pre/postprocessing transforms for structured data."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.common import (
    Sequential,
    Choice,
    Identity,
    StypeDispatch,
    TaskDispatch,
    ToNumerical,
    ShuffleColumns,
)
from sdm.processing.numerical import (
    Clip,
    ClipByQuantiles,
    ClipBySigma,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    Standardize,
    DropConstant,
)
from sdm.processing.categorical import (
    AlignCategories,
    ShuffleCategories,
    ImputeCategories,
)
from sdm.processing.datetime import EncodeCalendar
from sdm.processing.output import ReduceEstimators, Softmax
from sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
    "Sequential",
    "Choice",
    "Identity",
    "StypeDispatch",
    "TaskDispatch",
    "ToNumerical",
    "ShuffleColumns",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeCategories",
    "Clip",
    "ClipByQuantiles",
    "ClipBySigma",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "Standardize",
    "DropConstant",
    "EncodeCalendar",
    "ReduceEstimators",
    "Softmax",
    "Recipe",
]
