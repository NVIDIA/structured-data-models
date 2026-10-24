# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data modeling primitives."""

__lazy_modules__ = {
    "sdm.models",
    "sdm.evaluation",
    "sdm.explain",
}

from importlib.metadata import PackageNotFoundError, version

from sdm._constants import NaT
from sdm.optimization import optimize
from sdm.stype import Stype, StypeLike, infer_stypes
from sdm.task import Task, TaskLike
from sdm.tensor import (
    VarLenTensor,
    StringTensor,
    NullableTensor,
    CategoricalTensor,
    ColumnarTensor,
    TableTensor,
)
from sdm.ensemble import EnsembleData, EnsembleTable
from sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
)
from sdm.processing import Recipe
import sdm.models as models
import sdm.evaluation as evaluation
import sdm.explain as explain

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "NaT",
    "optimize",
    "Stype",
    "StypeLike",
    "Task",
    "TaskLike",
    "infer_stypes",
    "VarLenTensor",
    "StringTensor",
    "NullableTensor",
    "CategoricalTensor",
    "ColumnarTensor",
    "TableTensor",
    "EnsembleData",
    "EnsembleTable",
    "Relationship",
    "RelationalData",
    "TaskLink",
    "RelatedTables",
    "Recipe",
    "models",
    "evaluation",
    "explain",
    "__version__",
]
