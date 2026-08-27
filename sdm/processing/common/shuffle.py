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
        method: Literal["shift", "random"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[Tensor] = BufferList()
        self._host_permutations: tuple[tuple[int, ...], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, ...], ...]:
        r""":meta private:"""  # noqa: D415
        return self._host_permutations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._host_permutations = cast(tuple[tuple[int, ...], ...], state)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        permutations = []
        host_permutations = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            n_features = table.numerical.size(-1)
            device = table.numerical.device
            if n_features <= 1:
                permutation = torch.arange(n_features, device=device)
            elif self.method == "shift":
                offset = torch.randint(
                    n_features,
                    (1,),
                    generator=generator,
                    device=device,
                )
                permutation = (
                    torch.arange(n_features, device=device) + offset
                ) % n_features
            else:
                assert self.method == "random"
                permutation = torch.randperm(
                    n_features,
                    generator=generator,
                    device=device,
                )
            permutations.append(permutation)
            host_permutations.append(tuple(permutation.tolist()))
        self._permutations = BufferList(permutations)
        self._host_permutations = tuple(host_permutations)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutations) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._permutations)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )
        tables: list[TableTensor] = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            permutation = self._permutations[member_id]
            host_permutation = self._host_permutations[member_id]
            out = table.__class__(
                columns={
                    Stype.numerical: tuple(
                        table.columns[Stype.numerical][i]
                        for i in host_permutation
                    )
                },
                numerical=table.numerical.index_select(-1, permutation),
            )
            tables.append(
                cast(
                    TableTensor,
                    torch.cat((table.drop_stypes(Stype.numerical), out), dim=-1),
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutations) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._permutations)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )
        tables: list[TableTensor] = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            host_permutation = self._host_permutations[member_id]
            inverse_host_permutation = [0] * len(host_permutation)
            for destination, source in enumerate(host_permutation):
                inverse_host_permutation[source] = destination
            permutation = self._permutations[member_id].argsort()
            out = table.__class__(
                columns={
                    Stype.numerical: tuple(
                        table.columns[Stype.numerical][i]
                        for i in inverse_host_permutation
                    )
                },
                numerical=table.numerical.index_select(-1, permutation),
            )
            tables.append(
                cast(
                    TableTensor,
                    torch.cat((table.drop_stypes(Stype.numerical), out), dim=-1),
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
