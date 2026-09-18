# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data modeling primitives."""

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
from sdm.ensemble import EnsembleTable
from sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
)
from sdm.processing import Recipe
from sdm import models, evaluation, explain

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
