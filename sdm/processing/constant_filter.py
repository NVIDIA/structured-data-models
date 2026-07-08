import math
from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor

ConstantFilterMethod = Literal["unique", "variance"]


class ConstantFilter(Processor):
    """Remove non-informative numerical columns learned during fit.

    With ``method="unique"``, NaN values count as one distinct category: a
    column containing one finite value and NaN has two unique values, while an
    all-NaN column has one. When the number of samples is less than or equal to
    ``threshold``, all columns are preserved.

    With ``method="variance"``, columns are retained when their sample
    standard deviation is greater than ``tolerance``. This method requires
    floating-point input. Columns containing NaN have NaN standard deviation
    and are removed.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Filtering rule. ``"unique"`` uses distinct-value counts;
            ``"variance"`` uses sample standard deviation.
        threshold: With ``method="unique"``, columns with at most this many
            unique values are removed.
        tolerance: With ``method="variance"``, columns with sample standard
            deviation at most this value are removed.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: ConstantFilterMethod = "unique",
        *,
        threshold: int = 1,
        tolerance: float = 1e-6,
    ) -> None:
        super().__init__()
        if method not in {"unique", "variance"}:
            raise ValueError("method must be one of 'unique' or 'variance'")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.method = method
        self.threshold = threshold
        self.tolerance = tolerance
        self.columns_to_keep: tuple[str, ...] = ()

    def _fit(self, input: TableTensor) -> None:
        numerical = input.numerical
        n_samples = math.prod(input.size()[:-1])
        data = numerical.reshape(n_samples, numerical.size(-1))

        if self.method == "unique":
            keep = _keep_unique(data, threshold=self.threshold)
        else:
            if not data.is_floating_point():
                raise TypeError(
                    "Expected floating-point numerical data for "
                    "method='variance'"
                )
            keep = data.std(dim=0) > self.tolerance

        # Mapping a tensor mask back to schema names requires one device sync.
        indices = keep.nonzero().flatten().tolist()
        columns = input.columns[Stype.numerical]
        self.columns_to_keep = tuple(columns[index] for index in indices)

    def _transform(self, input: TableTensor) -> TableTensor:
        """Drop columns rejected by the fitted filtering rule."""
        if len(self.columns_to_keep) == input.numerical.size(-1):
            return input
        return input.select_columns(self.columns_to_keep)


def _keep_unique(input: Tensor, *, threshold: int) -> Tensor:
    n_samples, n_columns = input.size()
    if n_samples <= threshold or threshold == 0:
        return input.new_ones((n_columns,), dtype=torch.bool)

    if threshold == 1:
        # [N, C] compared with [1, C] -> one keep decision per column.
        same_as_first = _equal_with_nan(input, input[:1])
        return (~same_as_first).any(dim=0)

    # [N, C] -> [N - 1, C] adjacent equality after column-wise sort.
    values = input.sort(dim=0).values
    adjacent_equal = _equal_with_nan(values[1:], values[:-1])
    unique_counts = (~adjacent_equal).sum(dim=0) + 1
    return unique_counts > threshold


def _equal_with_nan(left: Tensor, right: Tensor) -> Tensor:
    equal = left == right
    if left.is_floating_point():
        equal = equal | (left.isnan() & right.isnan())
    return equal
