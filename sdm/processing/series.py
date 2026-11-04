# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from torch import Tensor

from sdm.processing import EnsembleProcessor


class SeriesEnsembleProcessor(EnsembleProcessor):
    r"""Base processor for consecutive, equally spaced time-series data."""

    _offset: Tensor | None

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_offset", None, persistent=False)
