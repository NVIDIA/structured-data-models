# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processors for structured data tables."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.target import TargetAwareProcessor
from sdm.processing.series import SeriesProcessor
from sdm.processing.common import (
    Identity,
    Callable,
    Sequential,
    DropStypes,
    StypeDispatch,
    TaskDispatch,
    TableDispatch,
    Choice,
    ToNumerical,
    ShuffleColumns,
    SelectColumns,
)
from sdm.processing.text import TFIDF, SentenceTransformer
from sdm.processing.numerical import (
    Cast,
    Clip,
    ClipQuantiles,
    ClipSigma,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    RankGaussian,
    Standardize,
    RobustScale,
    MinMaxScale,
    FlipSign,
    Asinh,
    DropConstantColumns,
    PCA,
    RandomProjection,
    Detrend,
    Interpolate,
)
from sdm.processing.categorical import (
    AlignCategories,
    ShuffleCategories,
    ImputeMode,
    ToOneHot,
    AddCategoryCounts,
)
from sdm.processing.datetime import AddCalendarFields
from sdm.processing.output import (
    AverageEstimators,
    Softmax,
    SortQuantiles,
    ClipNonnegative,
)
from sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
    "TargetAwareProcessor",
    "SeriesProcessor",
    "Identity",
    "Callable",
    "Sequential",
    "DropStypes",
    "StypeDispatch",
    "TaskDispatch",
    "TableDispatch",
    "Choice",
    "ToNumerical",
    "ShuffleColumns",
    "SelectColumns",
    "TFIDF",
    "SentenceTransformer",
    "Cast",
    "Clip",
    "ClipQuantiles",
    "ClipSigma",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "RankGaussian",
    "Standardize",
    "RobustScale",
    "MinMaxScale",
    "FlipSign",
    "Asinh",
    "DropConstantColumns",
    "PCA",
    "RandomProjection",
    "Detrend",
    "Interpolate",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeMode",
    "ToOneHot",
    "AddCategoryCounts",
    "AddCalendarFields",
    "AverageEstimators",
    "Softmax",
    "SortQuantiles",
    "ClipNonnegative",
    "Recipe",
]
