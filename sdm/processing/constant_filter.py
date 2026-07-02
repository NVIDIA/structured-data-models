import math

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import CategoricalTensor, TableTensor


class ConstantFilter(Processor):
    """Remove columns with too few unique values learned during fit.

    Args:
        threshold: Columns with at most this many unique values are removed.
            When the number of samples is less than or equal to ``threshold``,
            all columns are preserved.
    """

    input_scope = "table"

    def __init__(self, threshold: int = 1) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self.columns_to_keep: dict[Stype, tuple[str, ...]] = {}

    def _fit(self, input: Tensor) -> None:
        if not isinstance(input, TableTensor):
            raise TypeError(
                "Expected ConstantFilter input to be a TableTensor "
                f"(got '{type(input).__name__}')"
            )

        n_samples = math.prod(input.size()[:-1])
        columns_to_keep: dict[Stype, tuple[str, ...]] = {}

        for stype, block in input.items():
            columns = input.columns[stype]
            if n_samples <= self.threshold:
                columns_to_keep[stype] = columns
                continue

            data = _as_column_data(block).reshape(n_samples, block.size(-1))
            keep = []
            for index, column in enumerate(columns):
                if _unique_count(data[:, index]) > self.threshold:
                    keep.append(column)
            columns_to_keep[stype] = tuple(keep)

        self.columns_to_keep = columns_to_keep

    def forward(self, input: Tensor) -> Tensor:
        """Drop columns that were constant in the fitted data."""
        if not isinstance(input, TableTensor):
            raise TypeError(
                "Expected ConstantFilter input to be a TableTensor "
                f"(got '{type(input).__name__}')"
            )

        columns = tuple(
            column
            for stype_columns in self.columns_to_keep.values()
            for column in stype_columns
        )
        if len(columns) == input.size(-1):
            return input
        return input.select_columns(columns)


def _as_column_data(input: Tensor) -> Tensor:
    if isinstance(input, CategoricalTensor):
        return input.as_tensor()
    return input


def _unique_count(input: Tensor) -> int:
    if not torch.is_floating_point(input):
        return torch.unique(input).numel()

    finite = input[~torch.isnan(input)]
    count = torch.unique(finite).numel()
    if torch.isnan(input).any():
        count += 1
    return count
