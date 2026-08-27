import math
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable


class CrossFeatures(EnsembleProcessor):
    r"""Append products of randomly selected numerical feature pairs.

    Pair selection is fitted independently for each ensemble member and
    replayed for query tables. Pass ``generator`` to ``fit()`` to make the
    selection reproducible.

    Args:
        num_crosses: Number of product features to append. ``"sqrt"`` appends
            the integer square root of the input feature count, bounded by the
            number of distinct feature pairs.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        num_crosses: int | Literal["sqrt"],
    ) -> None:
        super().__init__()
        if isinstance(num_crosses, int) and num_crosses < 0:
            raise ValueError("num_crosses must be non-negative")
        self.num_crosses = num_crosses
        self._pairs: BufferList[Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pairs = [
            self._draw_pairs(
                ensemble_table.table(member_id).numerical.size(-1),
                device=ensemble_table.table(member_id).device,
                generator=generator,
            )
            for member_id in range(ensemble_table.num_members)
        ]
        self._pairs = BufferList(pairs)

    def _draw_pairs(
        self,
        num_features: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        max_crosses = num_features * (num_features - 1) // 2
        num_crosses = (
            math.isqrt(num_features)
            if self.num_crosses == "sqrt"
            else self.num_crosses
        )
        num_crosses = min(num_crosses, max_crosses)
        if num_crosses == 0:
            return torch.empty((0, 2), device=device, dtype=torch.long)
        sampling_device = generator.device if generator is not None else device
        if num_crosses == max_crosses:
            pairs = torch.combinations(
                torch.arange(num_features, device=sampling_device),
                r=2,
            )
            return pairs.to(device)
        if num_crosses * 2 >= max_crosses:
            pairs = torch.combinations(
                torch.arange(num_features, device=sampling_device),
                r=2,
            )
            permutation = torch.randperm(
                max_crosses,
                generator=generator,
                device=sampling_device,
            )
            return pairs.index_select(0, permutation[:num_crosses]).to(device)

        pairs = torch.empty((0, 2), device=sampling_device, dtype=torch.long)
        while pairs.size(0) < num_crosses:
            remaining = num_crosses - pairs.size(0)
            candidates = (
                torch.randint(
                    num_features,
                    (remaining * 2, 2),
                    generator=generator,
                    device=sampling_device,
                )
                .sort(dim=-1)
                .values
            )
            candidates = candidates[candidates[:, 0] != candidates[:, 1]]
            pairs = torch.unique(torch.cat((pairs, candidates)), dim=0)
        permutation = torch.randperm(
            pairs.size(0),
            generator=generator,
            device=sampling_device,
        )
        return pairs.index_select(0, permutation[:num_crosses]).to(device)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._pairs) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._pairs)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )

        tables = [
            self._transform_member(ensemble_table.table(member_id), member_id)
            for member_id in range(ensemble_table.num_members)
        ]
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _transform_member(
        self,
        table: TableTensor,
        member_id: int,
    ) -> TableTensor:
        pairs = self._pairs[member_id]
        if pairs.size(0) == 0:
            return table

        left, right = pairs.unbind(dim=-1)
        crossed = table.numerical.index_select(-1, left) * (
            table.numerical.index_select(-1, right)
        )
        existing = set(table.column_names)
        columns = []
        index = 0
        while len(columns) < pairs.size(0):
            column = f"cross_{index}"
            if column not in existing:
                columns.append(column)
                existing.add(column)
            index += 1
        out = TableTensor(
            columns={Stype.numerical: tuple(columns)},
            numerical=crossed,
        )
        return cast(TableTensor, torch.cat((table, out), dim=-1))

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_crosses={self.num_crosses!r})"
        )
