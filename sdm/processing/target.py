# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import TableLike
from sdm.processing import Processor


class TargetAwareProcessor(Processor):
    r"""Base processor with access to unprocessed targets."""

    def __init__(self) -> None:
        super().__init__()
        self._target: TableLike | None = None
