# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import TableTensor
from sdm.processing import EnsembleProcessor


class TargetAwareEnsembleProcessor(EnsembleProcessor):
    r"""Base processor with access to unprocessed context targets.

    During recipe execution, ``_target`` contains the original target table,
    preserving its input batch dimensions.
    """

    _target: TableTensor | None

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_target", None, persistent=False)
