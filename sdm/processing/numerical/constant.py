from typing import Literal

import torch

from sdm import Stype
from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.tensor import TableTensor

DropConstantMethod = Literal["unique", "variance"]


class DropConstant(Processor):
    """Remove non-informative numerical columns learned during fit.

    With ``method="unique"``, columns are retained when they have more than
    ``threshold`` distinct values. When the number of samples is less than or
    equal to ``threshold``, all columns are preserved.

    With ``method="variance"``, columns are retained when their sample
    standard deviation is greater than ``tolerance``. Non-floating input is
    promoted to the default floating-point dtype for this calculation.

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
        method: DropConstantMethod = "unique",
        *,
        threshold: int | None = None,
        tolerance: float | None = None,
    ) -> None:
        super().__init__()
        if method not in {"unique", "variance"}:
            raise ValueError("method must be 'unique' or 'variance'")

        if method == "unique" and tolerance is not None:
            raise ValueError("tolerance must be None when method is 'unique'")

        if method == "variance" and threshold is not None:
            raise ValueError(
                "threshold must be None when method is 'variance'"
            )

        if threshold is not None and threshold <= 0:
            raise ValueError("threshold must be positive")

        if tolerance is not None and tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.method = method
        self.threshold = 1 if threshold is None else threshold
        self.tolerance = 1e-6 if tolerance is None else tolerance
        self._columns_to_keep: tuple[str, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        data = table.numerical

        if self.method == "variance":
            keep = _as_float(data).std(dim=0) > self.tolerance
        # Preserve the schema when too few rows can exceed the threshold.
        elif data.size(0) <= self.threshold:
            keep = data.new_ones((data.size(-1),), dtype=torch.bool)
        elif self.threshold == 1:
            # Any mismatch with the first row proves a second unique value.
            first = data[:1]
            different = data != first
            keep = different.any(dim=0)
        else:
            # A sorted column with k unique values has k - 1 transitions.
            values = data.sort(dim=0).values
            left, right = values[1:], values[:-1]
            changed = left != right
            keep = changed.sum(dim=0) >= self.threshold

        indices = keep.nonzero().flatten().tolist()
        columns = table.columns[Stype.numerical]
        self._columns_to_keep = tuple(columns[index] for index in indices)

    def _transform(self, table: TableTensor) -> TableTensor:
        """Drop columns rejected by the fitted filtering rule."""
        columns = table.columns[Stype.numerical]
        if self._columns_to_keep == columns:
            return table
        return table.select_columns(self._columns_to_keep)
