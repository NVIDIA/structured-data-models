from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.ensemble import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable, TableTensor


class _ColumnPermutation(torch.nn.Module):
    """Register one column permutation as PyTorch module state."""

    indices: Tensor

    def __init__(self, indices: Tensor) -> None:
        super().__init__()
        self.register_buffer("indices", indices)


class ShuffleColumns(EnsembleProcessor, EnsembleInvertibleMixin):
    """Permute the numerical feature columns.

    The permutation is drawn when the processor is fitted; pass
    ``generator`` to ``fit()`` to make it reproducible. Convert
    non-numerical feature stypes before this step, for example with
    :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            columns by a drawn offset, and ``"random"`` permutes the columns
            with a drawn permutation.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations = torch.nn.ModuleList()
        self._member_permutation_ids: tuple[int, ...] = ()

    @property
    def permutation(self) -> Tensor:
        """Return the fitted permutation for a single table."""
        if len(self._member_permutation_ids) != 1:
            raise RuntimeError(
                "'ShuffleColumns' has no single fitted permutation."
            )
        permutation_id = self._member_permutation_ids[0]
        state = cast(_ColumnPermutation, self._permutations[permutation_id])
        return state.indices

    def _draw_permutation(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        n_features = table.numerical.size(-1)
        device = table.numerical.device
        if n_features <= 1:
            return torch.arange(n_features, device=device)
        if self.method == "shift":
            offset = torch.randint(
                n_features,
                (1,),
                generator=generator,
                device=device,
            )
            return (
                torch.arange(n_features, device=device) + offset
            ) % n_features
        return torch.randperm(
            n_features,
            generator=generator,
            device=device,
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._permutations = torch.nn.ModuleList()
        member_permutation_ids = []
        fitted: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}

        for member_id in range(ensemble_table.num_members):
            permutation = self._draw_permutation(
                ensemble_table.table(member_id),
                generator=generator,
            )
            key = (
                ensemble_table._locations[member_id],
                tuple(permutation.tolist()),
            )
            permutation_id = fitted.get(key)
            if permutation_id is None:
                permutation_id = len(self._permutations)
                fitted[key] = permutation_id
                self._permutations.append(_ColumnPermutation(permutation))
            member_permutation_ids.append(permutation_id)

        self._member_permutation_ids = tuple(member_permutation_ids)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._apply_ensemble(ensemble_table, inverse=False)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._apply_ensemble(ensemble_table, inverse=True)

    def _apply_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        inverse: bool,
    ) -> EnsembleTable:
        if len(self._member_permutation_ids) != ensemble_table.num_members:
            operation = "inverse transform" if inverse else "transform"
            raise RuntimeError(
                "ShuffleColumns must be fitted with the same number of "
                f"ensemble members before {operation}."
            )

        tables = []
        member_table_ids = []
        table_ids: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, permutation_id in enumerate(
            self._member_permutation_ids
        ):
            key = (ensemble_table._locations[member_id], permutation_id)
            table_id = table_ids.get(key)
            if table_id is None:
                state = cast(
                    _ColumnPermutation,
                    self._permutations[permutation_id],
                )
                permutation = state.indices
                if inverse:
                    permutation = permutation.argsort()
                table_id = len(tables)
                table_ids[key] = table_id
                member_table = ensemble_table.table(member_id)
                tables.append(self._permute(member_table, permutation))
            member_table_ids.append(table_id)

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        permutation: Tensor,
    ) -> TableTensor:
        indices = permutation.tolist()
        return table.__class__(
            columns={
                Stype.numerical.value: tuple(
                    table.columns[Stype.numerical][index] for index in indices
                )
            },
            numerical=table.numerical.index_select(-1, permutation),
        )
