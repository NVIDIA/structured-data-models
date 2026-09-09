# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data modeling primitives."""

from importlib.metadata import PackageNotFoundError, version

from sdm._constants import NaT
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
from sdm.relational import (
    Relationship,
    RelationalData,
    TaskLink,
    RelatedTables,
)
from sdm.processing import Recipe
from sdm import callbacks, models, evaluation, explain

try:
    __version__ = version("structured-data-models")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "NaT",
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
    "Relationship",
    "RelationalData",
    "TaskLink",
    "RelatedTables",
    "Recipe",
    "callbacks",
    "models",
    "evaluation",
    "explain",
    "__version__",
]
