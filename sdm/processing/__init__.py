"""Processors for structured data tables."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.ensemble import (
    EnsembleProcessor,
    EnsembleInvertibleMixin,
    EnsembleProcessorAdapter,
)
from sdm.processing.common import (
    Identity,
    Callable,
    Sequential,
    StypeDispatch,
    TaskDispatch,
    Choice,
    ToNumerical,
    ShuffleColumns,
    SelectColumns,
)
from sdm.processing.text.tfidf_text_embed import TfidfTextEmbed
from sdm.processing.numerical import (
    Clip,
    ClipQuantiles,
    ClipSigma,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    Standardize,
    DropConstantColumns,
    PCA,
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
    "EnsembleProcessor",
    "EnsembleProcessorAdapter",
    "EnsembleInvertibleMixin",
    "Identity",
    "Callable",
    "Sequential",
    "StypeDispatch",
    "TaskDispatch",
    "Choice",
    "ToNumerical",
    "ShuffleColumns",
    "SelectColumns",
    "TfidfTextEmbed",
    "Clip",
    "ClipQuantiles",
    "ClipSigma",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "Standardize",
    "DropConstantColumns",
    "PCA",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeMode",
    "AddCalendarFields",
    "ReduceEstimators",
    "Softmax",
    "Recipe",
]
