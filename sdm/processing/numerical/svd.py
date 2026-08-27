import math
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor, Identity
from sdm.tensor import EnsembleTable


class TruncatedSVD(EnsembleProcessor):
    r"""Append selected features from an uncentered low-rank projection.

    The preprocessor is fitted on context data and must return a numerical-only
    table. One component pool is fitted per compatible source group. Each
    logical ensemble member selects components from that shared pool and
    replays the same selection for query data. Pool size is bounded by the
    smaller of the fitted row and prepared feature counts, minus one.

    Half-precision inputs use float32 for the SVD and projection math before
    generated features are cast back to the original numerical dtype.

    Args:
        num_components: Number of generated features per member. ``"sqrt"``
            uses the integer square root of the original input column count.
        pool_size: Number of components fitted in each shared pool. ``None``
            uses ``num_components``. ``"sum"`` uses the sum requested by all
            logical members reaching this processor.
        preprocessor: Processor composition used only to prepare the SVD
            matrix. The fitted transformation is replayed for query data.
    """

    handles_stypes = frozenset(Stype)
    requires_fit = True

    def __init__(
        self,
        num_components: int | Literal["sqrt"],
        *,
        pool_size: int | Literal["sum"] | None = None,
        preprocessor: object = None,
    ) -> None:
        super().__init__()
        if isinstance(num_components, int) and num_components < 0:
            raise ValueError("num_components must be non-negative")
        if isinstance(pool_size, int) and pool_size < 0:
            raise ValueError("pool_size must be non-negative")
        self.num_components = num_components
        self.pool_size = pool_size
        self.preprocessor = EnsembleProcessor.as_processor(
            Identity() if preprocessor is None else preprocessor
        )
        self._components: BufferList[Tensor] = BufferList()
        self._selections: BufferList[Tensor] = BufferList()
        self._locations: tuple[tuple[int, int], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, int], ...]:
        r""":meta private:"""  # noqa: D415
        return self._locations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._locations = cast(tuple[tuple[int, int], ...], state)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        prepared = self.preprocessor.fit_transform_ensemble(
            ensemble_table, generator=generator
        )
        self._fit_prepared(ensemble_table, prepared, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        prepared = self.preprocessor.fit_transform_ensemble(
            ensemble_table, generator=generator
        )
        self._fit_prepared(ensemble_table, prepared, generator=generator)
        return self._append(ensemble_table, prepared)

    def _fit_prepared(
        self,
        original: EnsembleTable,
        prepared: EnsembleTable,
        *,
        generator: torch.Generator | None,
    ) -> None:
        self._check_prepared(prepared)
        requested = [
            math.isqrt(original.table(i).size(-1))
            if self.num_components == "sqrt"
            else self.num_components
            for i in range(original.num_members)
        ]
        components = []
        for group_id, group in enumerate(prepared):
            group_requested = [
                requested[i]
                for i, (fitted_group_id, _) in enumerate(prepared._locations)
                if fitted_group_id == group_id
            ]
            requested_pool = (
                sum(group_requested)
                if self.pool_size == "sum"
                else self.pool_size
            )
            if requested_pool is None:
                requested_pool = max(group_requested, default=0)
            if requested_pool < max(group_requested, default=0):
                raise ValueError("pool_size must cover num_components")
            x = group.numerical
            components.append(
                _truncated_svd(
                    x,
                    num_components=requested_pool,
                    generator=generator,
                )
            )
        self._components = BufferList(components)
        self._locations = prepared._locations

        selections = []
        for member_id, (group_id, _) in enumerate(self._locations):
            pool = self._components[group_id].size(-1)
            size = min(requested[member_id], pool)
            device = prepared.table(member_id).device
            sampling_device = (
                generator.device if generator is not None else device
            )
            selections.append(
                torch.randperm(
                    pool,
                    generator=generator,
                    device=sampling_device,
                )[:size].to(device)
            )
        self._selections = BufferList(selections)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._locations) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._locations)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )
        prepared = self.preprocessor.transform_ensemble(ensemble_table)
        self._check_prepared(prepared)
        return self._append(ensemble_table, prepared)

    def _append(
        self,
        original: EnsembleTable,
        prepared: EnsembleTable,
    ) -> EnsembleTable:
        tables = []
        for member_id, (group_id, position) in enumerate(self._locations):
            table = original.table(member_id)
            x = prepared.table(member_id).numerical
            components = self._components[group_id][position]
            components = components.index_select(
                -1, self._selections[member_id]
            )
            projected = (x.to(components.dtype) @ components).to(
                table.numerical.dtype
            )
            existing = set(table.column_names)
            columns = []
            index = 0
            while len(columns) < projected.size(-1):
                column = f"svd_{index}"
                if column not in existing:
                    columns.append(column)
                    existing.add(column)
                index += 1
            out = TableTensor(
                columns={Stype.numerical: tuple(columns)},
                numerical=projected,
            )
            tables.append(cast(TableTensor, torch.cat((table, out), dim=-1)))
        return EnsembleTable.from_tables(
            tables=tables, member_table_ids=range(len(tables))
        )

    @staticmethod
    def _check_prepared(prepared: EnsembleTable) -> None:
        if any(
            not group.active_stypes <= {Stype.numerical} for group in prepared
        ):
            raise ValueError("preprocessor must return numerical-only tables")


def _truncated_svd(
    x: Tensor,
    *,
    num_components: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Return a compact randomized basis for the right singular subspace."""
    dtype = (
        torch.float32
        if x.dtype in {torch.float16, torch.bfloat16}
        else x.dtype
    )
    x = x.to(dtype)
    rank = max(0, min(x.size(-2), x.size(-1)) - 1)
    size = min(num_components, rank)
    if size == 0:
        return x.new_empty((*x.size()[:-2], x.size(-1), 0))

    range_size = min(rank, size + 10)
    sampling_device = generator.device if generator is not None else x.device
    projection = torch.randn(
        (*x.size()[:-2], x.size(-1), range_size),
        generator=generator,
        device=sampling_device,
        dtype=x.dtype,
    ).to(x.device)
    basis = torch.linalg.qr(x @ projection, mode="reduced").Q
    for _ in range(5):
        basis = torch.linalg.qr(
            x @ (x.transpose(-2, -1) @ basis), mode="reduced"
        ).Q
    reduced = basis.transpose(-2, -1) @ x
    _, _, vh = torch.linalg.svd(reduced, full_matrices=False)
    return vh[..., :size, :].transpose(-2, -1)
