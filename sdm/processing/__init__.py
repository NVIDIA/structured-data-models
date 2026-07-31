"""Processors for structured data tables."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.common import (
    Identity,
    Callable,
    Sequential,
    StypeDispatch,
    TaskDispatch,
    Choice,
    ToNumerical,
    ShuffleColumns,
)
from sdm.processing.text import TfidfTextEmbed, ModelTextEmbed
from sdm.processing.numerical import (
    Clip,
    ClipQuantiles,
    ClipSigma,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    Standardize,
    DropConstantColumns,
    SliceColumns,
)
from sdm.processing.categorical import (
    AlignCategories,
    ShuffleCategories,
    ImputeMode,
)
from sdm.processing.datetime import AddCalendarFields
from sdm.processing.output import ReduceEstimators, Softmax
from sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
    "Identity",
    "Callable",
    "Sequential",
    "StypeDispatch",
    "TaskDispatch",
    "Choice",
    "ToNumerical",
    "ShuffleColumns",
    "TfidfTextEmbed",
    "ModelTextEmbed",
    "Clip",
    "ClipQuantiles",
    "ClipSigma",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "Standardize",
    "DropConstantColumns",
    "SliceColumns",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeMode",
    "AddCalendarFields",
    "ReduceEstimators",
    "Softmax",
    "Recipe",
]
