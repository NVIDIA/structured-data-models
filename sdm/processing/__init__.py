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
    "Clip",
    "QuantileClip",
    "CategoryShuffle",
    "ConstantFilter",
    "EncodeDatetime",
    "FeaturePermute",
    "Identity",
    "MeanImpute",
    "PCA",
    "Power",
    "SliceFeatures",
    "Quantile",
    "SigmaClip",
    "EnsembleReduce",
    "SoftmaxTemperature",
    "StandardScale",
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
    "AddCalendarFields",
    "ReduceEstimators",
    "Softmax",
    "Recipe",
]
