from typing import Literal

import torch

from sdm import Stype
from sdm.processing._utils import _as_float
from sdm.processing.base import VariableSchemaProcessor
from sdm.tensor import TableTensor

DropConstantColumnsMethod = Literal["unique", "variance"]


class DropConstantColumns(VariableSchemaProcessor):
    """Remove non-informative numerical columns learned during fit.

    With ``method="unique"``, columns are retained when they have more than
    ``threshold`` distinct values. When the number of samples is less than or
    equal to ``threshold``, all columns are preserved.

    With ``method="variance"``, columns are retained when their sample
    standard deviation is greater than ``tolerance``. Non-floating input is
    promoted to the default floating-point dtype for this calculation.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.
    :meth:`~sdm.processing.Processor.fit` expects data with shape ``[R, C]``.
    :meth:`~sdm.processing.VariableSchemaProcessor.fit_batch` expects data with
    shape ``[B, R, C]`` and learns a separate selection for each of the ``B``
    representations. ``R`` is the number of rows and ``C`` is the number of
    numerical columns.

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
        self._columns_to_keep: tuple[tuple[str, ...], ...] = ()

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

    def _fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if table.dim() != 3:
            raise ValueError(
                "'DropConstantColumns.fit_batch' expects shape [B, R, C]."
            )
        keep = self._keep_mask(table.numerical)
        columns = table.columns[Stype.numerical]
        self._columns_to_keep = tuple(
            tuple(
                columns[index] for index in mask.nonzero().flatten().tolist()
            )
            for mask in keep
        )

    def _transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        if table.dim() != 3:
            raise ValueError(
                "'DropConstantColumns.transform_batch' expects shape "
                "[B, R, C]."
            )
        if table.size(0) != len(self._columns_to_keep):
            raise ValueError(
                "Expected the fitted number of representations "
                f"(got {table.size(0)})."
            )
        columns = table.columns[Stype.numerical]
        output = []
        for index, columns_to_keep in enumerate(self._columns_to_keep):
            representation = table[index]
            if columns_to_keep != columns:
                representation = representation.select_columns(columns_to_keep)
            output.append(representation)
        return tuple(output)
