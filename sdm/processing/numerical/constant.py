from typing import Literal, cast

import torch

from sdm import Stype
from sdm.processing._utils import _as_float
from sdm.processing.ensemble import EnsembleProcessor
from sdm.tensor import EnsembleTable, TableTensor

DropConstantColumnsMethod = Literal["unique", "variance"]


class DropConstantColumns(EnsembleProcessor):
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
    Ensemble query members must preserve the member order used during fitting.

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
        self.processors = torch.nn.ModuleList()
        self._member_processor_ids: tuple[int, ...] = ()

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

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._fit(table, generator=generator)
        return self._transform(table)

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.processors = torch.nn.ModuleList()
        representations = []
        member_processor_ids = []
        fitted: dict[tuple[int, int], int] = {}

        for member_id in range(table.num_members):
            location = table.member_location(member_id)
            processor_id = fitted.get(location)
            if processor_id is None:
                processor = self.__class__(
                    method=self.method,
                    threshold=(
                        self.threshold if self.method == "unique" else None
                    ),
                    tolerance=(
                        self.tolerance if self.method == "variance" else None
                    ),
                )
                transformed = processor.fit_transform(
                    table.representation(member_id),
                    generator=generator,
                )
                processor_id = len(representations)
                fitted[location] = processor_id
                self.processors.append(processor)
                representations.append(transformed)
            member_processor_ids.append(processor_id)

        self._member_processor_ids = tuple(member_processor_ids)
        return EnsembleTable.from_representations(
            representations=representations,
            member_representation_ids=self._member_processor_ids,
        )

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        if len(self._member_processor_ids) != table.num_members:
            raise RuntimeError(
                "DropConstantColumns must be fitted with the same number of "
                "ensemble members before transform."
            )

        representations = []
        member_representation_ids = []
        transformed: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, processor_id in enumerate(self._member_processor_ids):
            key = (table.member_location(member_id), processor_id)
            representation_id = transformed.get(key)
            if representation_id is None:
                processor = cast(
                    DropConstantColumns,
                    self.processors[processor_id],
                )
                representation_id = len(representations)
                transformed[key] = representation_id
                representations.append(
                    processor.transform(table.representation(member_id))
                )
            member_representation_ids.append(representation_id)

        return EnsembleTable.from_representations(
            representations=representations,
            member_representation_ids=member_representation_ids,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Drop columns rejected by the fitted filtering rule."""
        if len(self.processors) > 0:
            raise RuntimeError(
                "'DropConstantColumns' was fitted for an ensemble; use "
                "'transform_ensemble' instead of 'transform'."
            )
        columns = table.columns[Stype.numerical]
        if self._columns_to_keep == columns:
            return table
        return table.select_columns(self._columns_to_keep)
