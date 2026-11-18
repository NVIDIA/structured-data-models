# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic tabular datasets."""

from sdm.prior.tabular.schema import Schema, SchemaSampler
from sdm.prior.tabular.scm import SCMPrior

__all__ = [
    "Schema",
    "SchemaSampler",
    "SCMPrior",
]
