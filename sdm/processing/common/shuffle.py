from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable


class ShuffleColumns(EnsembleProcessor, EnsembleInvertibleMixin):
    """Permute numerical feature columns and their names.

    Pass ``generator`` to ``fit()`` to make the permutation reproducible.
    Convert non-numerical feature stypes before this step, for example with
    :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            columns by a drawn offset, and ``"random"`` permutes the columns
            with a drawn permutation.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations = BufferList()
        self._orders: tuple[tuple[int, ...], ...] = ()

    @property
    def permutation(self) -> Tensor:
        """Return the fitted permutation for a single table."""
        if len(self._permutations) != 1:
            raise RuntimeError(
                "'ShuffleColumns' has no single fitted permutation."
            )
        return self._permutations[0]

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

        assert self.method == "random"
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
        permutations = []
        orders = []
        for member_id in range(ensemble_table.num_members):
            permutation = self._draw_permutation(
                ensemble_table.table(member_id),
                generator=generator,
            )
            permutations.append(permutation)
            orders.append(tuple(permutation.tolist()))
        self._permutations = BufferList(permutations)
        self._orders = tuple(orders)

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
        if len(self._permutations) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._permutations)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )

        tables: list[TableTensor] = []
        for member_id, (fitted_permutation, fitted_order) in enumerate(
            zip(self._permutations, self._orders, strict=True)
        ):
            permutation = (
                fitted_permutation.argsort() if inverse else fitted_permutation
            )
            order = tuple(permutation.tolist()) if inverse else fitted_order
            tables.append(
                self._permute(
                    ensemble_table.table(member_id),
                    permutation,
                    order,
                )
            )

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        permutation: Tensor,
        order: tuple[int, ...],
    ) -> TableTensor:
        out = table.__class__(
            columns={
                Stype.numerical: tuple(
                    table.columns[Stype.numerical][index] for index in order
                )
            },
            numerical=table.numerical.index_select(-1, permutation),
        )
        return cast(
            TableTensor,
            torch.cat((table.drop_stypes(Stype.numerical), out), dim=-1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
