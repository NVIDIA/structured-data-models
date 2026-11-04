# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from sdm import Stype, TableTensor
from sdm.processing import Processor


class SortQuantiles(Processor):
    r"""Sort quantile predictions in ascending order.

    Args:
        num_quantiles: Number of consecutive numerical columns per target in
            case of multi-target predictions.
            If ``None``, all numerical columns form one quantile group.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def __init__(self, num_quantiles: int | None = None) -> None:
        super().__init__()
        self.num_quantiles = num_quantiles

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        if self.num_quantiles is not None:
            numerical = numerical.unflatten(-1, (-1, self.num_quantiles))
        numerical = numerical.sort(dim=-1).values
        if self.num_quantiles is not None:
            numerical = numerical.flatten(-2)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        if self.num_quantiles is None:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_quantiles={self.num_quantiles})"
        )
