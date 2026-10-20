# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processors that mask numerical and categorical cells."""

from sdm.processing.missing.random import MaskRandom

__all__ = [
    "MaskRandom",
]
