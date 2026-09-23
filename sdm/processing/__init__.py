# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processors for structured data tables."""

from sdm.processing.base import Processor, InvertibleMixin
from sdm.processing.ensemble import (
    EnsembleProcessor,
    EnsembleInvertibleMixin,
)
from sdm.processing.common import (
    Identity,
    Callable,
    Sequential,
    DropStypes,
    StypeDispatch,
    TaskDispatch,
    TableDispatch,
    EnsembleProcessorAdapter,
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
    ClipSoft,
    ImputeMean,
    PowerTransform,
    QuantileTransform,
    Standardize,
    RobustScale,
    FlipSign,
    DropConstantColumns,
    PCA,
    RandomProjection,
)
from sdm.processing.categorical import (
    AlignCategories,
    ShuffleCategories,
    ImputeMode,
    AddCategoryCounts,
)
from sdm.processing.datetime import AddCalendarFields
from sdm.processing.output import AverageEstimators, Softmax, SortQuantiles
from sdm.processing.recipe import Recipe

__all__ = [
    "Processor",
    "InvertibleMixin",
    "EnsembleInvertibleMixin",
    "EnsembleProcessorAdapter",
    "Identity",
    "Callable",
    "Sequential",
    "DropStypes",
    "StypeDispatch",
    "TaskDispatch",
    "TableDispatch",
    "EnsembleProcessor",
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
    "ClipSoft",
    "ImputeMean",
    "PowerTransform",
    "QuantileTransform",
    "Standardize",
    "RobustScale",
    "FlipSign",
    "DropConstantColumns",
    "PCA",
    "RandomProjection",
    "AlignCategories",
    "ShuffleCategories",
    "ImputeMode",
    "AddCategoryCounts",
    "AddCalendarFields",
    "AverageEstimators",
    "Softmax",
    "SortQuantiles",
    "Recipe",
]
