from typing import Literal

import torch
from typing_extensions import Self

from sdm import Stype
from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.processing.ensemble import VariableSchemaBatchMixin
from sdm.tensor import TableTensor

DropConstantColumnsMethod = Literal["unique", "variance"]


class DropConstantColumns(Processor, VariableSchemaBatchMixin):
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
        method: DropConstantColumnsMethod = "unique",
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
        self._batch_columns_to_keep: tuple[tuple[str, ...], ...] = ()

    def _keep_mask(self, data: torch.Tensor) -> torch.Tensor:
        if self.method == "variance":
            return _as_float(data).std(dim=-2) > self.tolerance
        if data.size(-2) <= self.threshold:
            return data.new_ones(
                (*data.shape[:-2], data.size(-1)),
                dtype=torch.bool,
            )
        if self.threshold == 1:
            return (data != data[..., :1, :]).any(dim=-2)

        values = data.sort(dim=-2).values
        changed = values[..., 1:, :] != values[..., :-1, :]
        return changed.sum(dim=-2) >= self.threshold

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        data = table.numerical
        keep = self._keep_mask(data)
        indices = keep.nonzero().flatten().tolist()
        columns = table.columns[Stype.numerical]
        self._columns_to_keep = tuple(columns[index] for index in indices)

    def fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit one column mask per leading variant.

        Args:
            table: Variant batch with shape ``[V, R, C]``.
            generator: Unused optional pseudorandom number generator.
        """
        del generator
        self._check_supported_stypes(table)
        if table.dim() != 3:
            raise ValueError(
                "'DropConstantColumns.fit_batch' expects shape [V, R, C]."
            )
        keep = self._keep_mask(table.numerical)
        columns = table.columns[Stype.numerical]
        self._batch_columns_to_keep = tuple(
            tuple(
                columns[index] for index in mask.nonzero().flatten().tolist()
            )
            for mask in keep
        )
        return self

    def transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        r"""Apply fitted variant masks and return one table per schema.

        Args:
            table: Variant batch with shape ``[V, R, C]``.
        """
        if table.size(0) != len(self._batch_columns_to_keep):
            raise ValueError(
                "Expected the fitted number of leading variants "
                f"(got {table.size(0)})."
            )
        return tuple(
            table[index].select_columns(columns)
            for index, columns in enumerate(self._batch_columns_to_keep)
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Drop columns rejected by the fitted filtering rule."""
        columns = table.columns[Stype.numerical]
        if self._columns_to_keep == columns:
            return table
        return table.select_columns(self._columns_to_keep)
