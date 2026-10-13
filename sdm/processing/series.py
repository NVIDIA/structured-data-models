# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from sdm.processing import EnsembleProcessor


class SeriesProcessor(EnsembleProcessor):
    r"""Base processor for consecutive, equally spaced time-series data."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_offset", None, persistent=False)
