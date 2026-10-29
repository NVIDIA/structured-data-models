# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class Asinh(Processor, InvertibleMixin):
    """Apply the inverse hyperbolic sine to numerical values.

    Inverse transformation applies the hyperbolic sine.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.asinh())

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical.sinh())
