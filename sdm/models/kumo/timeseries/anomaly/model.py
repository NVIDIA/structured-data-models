# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from typing import ClassVar

from sdm import Recipe, Stype
from sdm.models import Model


class KumoAnomaly(Model):
    r"""The time series anomaly detection foundation model from the
    `"NV-Tesseract-AD: Curriculum-Enhanced Conditional Diffusion for
    Multivariate Time-Series Anomaly Detection"
    <https://arxiv.org/abs/2610.12754>`__ paper.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return Recipe()
