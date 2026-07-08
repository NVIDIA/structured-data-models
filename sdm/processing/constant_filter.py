from typing import Literal

import torch

from sdm import Stype
from sdm.processing._utils import _as_float
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
    standard deviation is greater than ``tolerance``. Non-floating input is
    promoted to the default floating-point dtype for this calculation. Columns
    containing NaN have NaN standard deviation and are removed.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.
    Fitting expects data with shape ``[N, C]``, where ``N`` is the number of
    rows and ``C`` is the number of numerical columns. The learned selection
    can transform later tables with shape ``[..., C]``.

    Args:
        method: Filtering rule. ``"unique"`` uses distinct-value counts;
            ``"variance"`` uses sample standard deviation.
        threshold: With ``method="unique"``, columns with at most this many
            unique values are removed. Must be positive.
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
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.method = method
        self.threshold = threshold
        self.tolerance = tolerance
        self._columns_to_keep: tuple[str, ...] = ()

    def _fit(self, input: TableTensor) -> None:
        data = input.numerical
        n_rows = data.size(0)

        if self.method == "variance":
            keep = _as_float(data).std(dim=0) > self.tolerance
        # Preserve the schema when too few rows can exceed the threshold.
        elif n_rows <= self.threshold:
            keep = data.new_ones((data.size(-1),), dtype=torch.bool)
        elif self.threshold == 1:
            first = data[:1]
            different = data != first
            if data.is_floating_point():
                different &= ~(data.isnan() & first.isnan())
            keep = different.any(dim=0)
        else:
            # A sorted column with k unique values has k - 1 transitions.
            values = data.sort(dim=0).values
            left, right = values[1:], values[:-1]
            changed = left != right
            if data.is_floating_point():
                changed &= ~(left.isnan() & right.isnan())
            keep = changed.sum(dim=0) >= self.threshold

        indices = keep.nonzero().flatten().tolist()
        columns = input.columns[Stype.numerical]
        self._columns_to_keep = tuple(columns[index] for index in indices)

    def _transform(self, input: TableTensor) -> TableTensor:
        """Drop columns rejected by the fitted filtering rule."""
        columns = input.columns[Stype.numerical]
        if self._columns_to_keep == columns:
            return input
        return input.select_columns(self._columns_to_keep)
