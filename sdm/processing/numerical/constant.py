from typing import Literal

import torch

from sdm import Stype
from sdm.processing.ensemble import EnsembleProcessor
from sdm.tensor import EnsembleTable, TableTensor

DropConstantColumnsMethod = Literal["unique", "variance"]


class DropConstantColumns(EnsembleProcessor):
    """Remove non-informative numerical columns learned during fit.

    With ``method="unique"``, columns are retained when they have more than
    ``threshold`` distinct values. When the number of samples is less than or
    equal to ``threshold``, all columns are preserved.

    With ``method="variance"``, columns are retained when their sample
    standard deviation is greater than ``tolerance``.

    Only numerical columns are supported. Convert other feature stypes before
    this step, for example with :class:`~sdm.processing.ToNumerical`.
    Fitting expects data with shape ``[N, C]``, where ``N`` is the number of
    rows and ``C`` is the number of numerical columns. The learned selection
    can transform later tables with shape ``[..., C]``.
    Each ensemble member learns its own column selection; query members use
    the selection fitted for the corresponding member.

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
        self._columns_by_member: tuple[tuple[str, ...], ...] = ()

    def _keep_mask(self, data: torch.Tensor) -> torch.Tensor:
        if self.method == "variance":
            return data.std(dim=-2) > self.tolerance
        if self.method == "unique":
            # Preserve the schema when too few rows can exceed the threshold.
            if data.size(-2) <= self.threshold:
                return data.new_ones(
                    (*data.shape[:-2], data.size(-1)),
                    dtype=torch.bool,
                )
            if self.threshold == 1:
                # Any mismatch with the first row proves a second unique value.
                return (data != data[..., :1, :]).any(dim=-2)

            # A sorted column with k unique values has k - 1 transitions.
            values = data.sort(dim=-2).values
            changed = values[..., 1:, :] != values[..., :-1, :]
            return changed.sum(dim=-2) >= self.threshold
        raise AssertionError(f"Unexpected method {self.method!r}")

    @staticmethod
    def _columns_from_mask(
        columns: tuple[str, ...],
        keep: list[bool],
    ) -> tuple[str, ...]:
        return tuple(
            column
            for column, keep_column in zip(columns, keep, strict=True)
            if keep_column
        )

    @staticmethod
    def _select_columns(
        table: TableTensor,
        columns: tuple[str, ...],
    ) -> TableTensor:
        if table.columns[Stype.numerical] == columns:
            return table
        return table.select_columns(columns)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        columns = table.columns[Stype.numerical]
        keep = self._keep_mask(table.numerical).tolist()
        self._columns_by_member = (self._columns_from_mask(columns, keep),)

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._select_columns(table, self._columns_by_member[0])

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._fit(table, generator=generator)
        return self._transform(table)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        groups = tuple(ensemble_table)
        if sum(group.size(0) for group in groups) == 1:
            group = groups[0]
            columns = group.columns[Stype.numerical]
            keep = self._keep_mask(group.numerical)[0].tolist()
            selection = self._columns_from_mask(columns, keep)
            self._columns_by_member = (selection,) * ensemble_table.num_members
            return

        masks = ensemble_table.replace_groups(
            [
                TableTensor.from_tensor(
                    tensor=self._keep_mask(group.numerical)
                    .to(dtype=group.numerical.dtype)
                    .unsqueeze(-2),
                    columns=group.columns[Stype.numerical],
                )
                for group in groups
            ]
        )
        masks_by_schema: dict[
            tuple[tuple[str, ...], torch.device],
            list[tuple[int, torch.Tensor]],
        ] = {}
        for member_id in range(masks.num_members):
            table = masks.table(member_id)
            columns = table.columns[Stype.numerical]
            key = (columns, table.device)
            masks_by_schema.setdefault(key, []).append(
                (member_id, table.numerical[0].bool())
            )

        columns_by_member: list[tuple[str, ...]] = [()] * masks.num_members
        for (columns, _), member_masks in masks_by_schema.items():
            keep_by_member = torch.stack(
                [keep for _, keep in member_masks]
            ).tolist()
            for (member_id, _), keep in zip(
                member_masks,
                keep_by_member,
                strict=True,
            ):
                columns_by_member[member_id] = self._columns_from_mask(
                    columns,
                    keep,
                )
        self._columns_by_member = tuple(columns_by_member)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._columns_by_member) != ensemble_table.num_members:
            raise RuntimeError(
                "DropConstantColumns must be fitted with the same number of "
                "ensemble members before transform."
            )

        if (
            sum(group.size(0) for group in ensemble_table)
            == ensemble_table.num_members
        ):
            tables = [
                self._select_columns(
                    ensemble_table.table(member_id),
                    columns,
                )
                for member_id, columns in enumerate(self._columns_by_member)
            ]
            return EnsembleTable.from_tables(
                tables=tables,
                member_table_ids=range(ensemble_table.num_members),
            )

        member_ids_by_columns: dict[tuple[str, ...], list[int]] = {}
        for member_id, columns in enumerate(self._columns_by_member):
            member_ids_by_columns.setdefault(columns, []).append(member_id)

        if len(member_ids_by_columns) == 1:
            columns = next(iter(member_ids_by_columns))
            return ensemble_table.replace_groups(
                [
                    self._select_columns(group, columns)
                    for group in ensemble_table
                ]
            )

        outputs: dict[tuple[str, ...], EnsembleTable] = {}
        for columns, member_ids in member_ids_by_columns.items():
            selected = ensemble_table.select_members(member_ids)
            outputs[columns] = selected.replace_groups(
                [self._select_columns(group, columns) for group in selected]
            )

        tables = []
        member_ids = []
        next_member_id_by_columns: dict[tuple[str, ...], int] = {}
        for columns in self._columns_by_member:
            tables.append(outputs[columns])
            member_id = next_member_id_by_columns.get(columns, 0)
            member_ids.append(member_id)
            next_member_id_by_columns[columns] = member_id + 1

        return EnsembleTable.gather_members(
            tables=tables,
            member_ids=member_ids,
        )
