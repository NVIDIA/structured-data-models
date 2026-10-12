# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm.processing import EnsembleProcessor


class SeriesProcessor(EnsembleProcessor):
    r"""Base processor for ordered rows on a time-series timeline.

    :class:`SeriesProcessor` is resolved only during model execution.
    """

    def __init__(self) -> None:
        super().__init__()
        self._offset: int | None = None
