from typing import Literal, cast

import torch

from sdm import Stype, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable

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

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

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
        # TODO: Consider recording the fitted column names if transforms should
        # verify that the numerical schema and order match fit.
        self._kept_indices: tuple[tuple[int, ...], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, ...], ...]:
        r""":meta private:"""  # noqa: D415
        return self._kept_indices

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._kept_indices = cast(tuple[tuple[int, ...], ...], state)

    def _keep_mask(self, data: torch.Tensor) -> torch.Tensor:
        # [N, C] or [..., N, C] -> [C] or [..., C].
        if self.method == "variance":
            return data.std(dim=-2) > self.tolerance

        assert self.method == "unique"
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

    @staticmethod
    def _select_columns(
        table: TableTensor,
        kept_indices: tuple[int, ...],
    ) -> TableTensor:
        numerical_columns = table.columns[Stype.numerical]
        if len(kept_indices) == len(numerical_columns):
            return table
        kept_numerical_columns = tuple(
            numerical_columns[index] for index in kept_indices
        )
        columns = [
            column
            for stype, stype_columns in table.columns.items()
            for column in (
                kept_numerical_columns
                if stype == Stype.numerical
                else stype_columns
            )
        ]
        return table.select_columns(columns)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        keep = self._keep_mask(table.numerical).tolist()
        self._kept_indices = (
            tuple(index for index, kept in enumerate(keep) if kept),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._select_columns(table, self._kept_indices[0])

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
            keep = self._keep_mask(group.numerical)[0].tolist()
            kept_indices = tuple(
                index for index, kept in enumerate(keep) if kept
            )
            self._kept_indices = (kept_indices,) * ensemble_table.num_members
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
        masks_by_size_and_device: dict[
            tuple[int, torch.device],
            list[tuple[int, torch.Tensor]],
        ] = {}
        for member_id in range(masks.num_members):
            table = masks.table(member_id)
            key = (table.numerical.size(-1), table.device)
            masks_by_size_and_device.setdefault(key, []).append(
                (member_id, table.numerical[0].bool())
            )

        kept_indices_by_member: list[tuple[int, ...]]
        kept_indices_by_member = [()] * masks.num_members
        for member_masks in masks_by_size_and_device.values():
            keep_by_member = torch.stack(
                [keep for _, keep in member_masks]
            ).tolist()
            for (member_id, _), keep in zip(
                member_masks,
                keep_by_member,
                strict=True,
            ):
                kept_indices_by_member[member_id] = tuple(
                    index for index, kept in enumerate(keep) if kept
                )
        self._kept_indices = tuple(kept_indices_by_member)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._kept_indices) != ensemble_table.num_members:
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
                    kept_indices,
                )
                for member_id, kept_indices in enumerate(self._kept_indices)
            ]
            return EnsembleTable.from_tables(
                tables=tables,
                member_table_ids=range(ensemble_table.num_members),
            )

        member_ids_by_kept_indices: dict[tuple[int, ...], list[int]] = {}
        for member_id, kept_indices in enumerate(self._kept_indices):
            member_ids_by_kept_indices.setdefault(kept_indices, []).append(
                member_id
            )

        if len(member_ids_by_kept_indices) == 1:
            kept_indices = next(iter(member_ids_by_kept_indices))
            return ensemble_table.replace_groups(
                [
                    self._select_columns(group, kept_indices)
                    for group in ensemble_table
                ]
            )

        outputs: dict[tuple[int, ...], EnsembleTable] = {}
        for kept_indices, member_ids in member_ids_by_kept_indices.items():
            selected = ensemble_table.select_members(member_ids)
            outputs[kept_indices] = selected.replace_groups(
                [
                    self._select_columns(group, kept_indices)
                    for group in selected
                ]
            )

        tables = []
        member_ids = []
        next_member_id_by_kept_indices: dict[tuple[int, ...], int] = {}
        for kept_indices in self._kept_indices:
            tables.append(outputs[kept_indices])
            member_id = next_member_id_by_kept_indices.get(kept_indices, 0)
            member_ids.append(member_id)
            next_member_id_by_kept_indices[kept_indices] = member_id + 1

        return EnsembleTable.gather_members(
            tables=tables,
            member_ids=member_ids,
        )
