# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import Stype, TableLike
from sdm.processing import InvertibleMixin, Processor


class Asinh(Processor, InvertibleMixin):
    r"""Apply the inverse hyperbolic sine to numerical values."""

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def _transform(self, table: TableLike) -> TableLike:
        return table.replace_blocks(numerical=table.numerical.asinh())

    def _inverse_transform(self, table: TableLike) -> TableLike:
        return table.replace_blocks(numerical=table.numerical.sinh())
