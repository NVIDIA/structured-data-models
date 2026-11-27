# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing.base import TensorStep


class Identity(Processor, InvertibleMixin):
    r"""Return inputs unchanged."""

    handles_stypes = tuple(Stype)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _tensor_steps(self) -> tuple[TensorStep, ...]:
        return ()

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table
