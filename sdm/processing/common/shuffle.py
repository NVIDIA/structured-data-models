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
        method: Permutation strategy. ``"random"`` (the default) draws
            independent permutations, and ``"latin"`` draws coupled Latin
            permutations.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        method: Literal["random", "latin"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[Tensor] = BufferList()
        self._host_permutations: tuple[tuple[tuple[int, ...], ...], ...] = ()
        self._locations: tuple[tuple[int, int], ...] = ()
        self._schemas: tuple[tuple[str, ...], ...] = ()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        schema_ids: dict[tuple[torch.device, tuple[str, ...]], int] = {}
        schemas: list[tuple[str, ...]] = []
        devices: list[torch.device] = []
        members: list[list[int]] = []
        locations: list[tuple[int, int]] = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            schema = table.columns[Stype.numerical]
            key = (table.device, schema)
            schema_id = schema_ids.get(key)
            if schema_id is None:
                schema_id = len(schemas)
                schema_ids[key] = schema_id
                schemas.append(schema)
                devices.append(table.device)
                members.append([])
            position = len(members[schema_id])
            members[schema_id].append(member_id)
            locations.append((schema_id, position))

        permutations: list[Tensor]
        if self.method == "latin":
            permutations = []
            for base, rows, patterns in self._latin_states(
                tuple(schemas),
                tuple(tuple(ids) for ids in members),
                tuple(devices),
                generator=generator,
            ):
                if base.numel() == 0:
                    permutations.append(base.new_empty((patterns.numel(), 0)))
                    continue
                permutations.append(
                    base[(patterns[:, None] - rows[None, :]) % base.numel()]
                )
        else:
            by_schema: list[list[Tensor]] = [[] for _ in schemas]
            for member_id, (schema_id, _) in enumerate(locations):
                table = ensemble_table.table(member_id)
                n_features = table.numerical.size(-1)
                device = devices[schema_id]
                if n_features <= 1:
                    permutation = torch.arange(n_features, device=device)
                else:
                    assert self.method == "random"
                    permutation = torch.randperm(
                        n_features,
                        generator=generator,
                        device=device,
                    )
                by_schema[schema_id].append(permutation)
            permutations = [torch.stack(values) for values in by_schema]

        self._permutations = BufferList(permutations)
        self._host_permutations = tuple(
            tuple(tuple(order) for order in permutation.tolist())
            for permutation in permutations
        )
        self._locations = tuple(locations)
        self._schemas = tuple(schemas)

    def _latin_states(
        self,
        schemas: tuple[tuple[str, ...], ...],
        members: tuple[tuple[int, ...], ...],
        devices: tuple[torch.device, ...],
        *,
        generator: torch.Generator | None,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
        if len(schemas) == 1:
            n_features = len(schemas[0])
            device = devices[0]
            base, rows, pattern_order = (
                torch.randperm(
                    n_features,
                    generator=generator,
                    device=device,
                )
                for _ in range(3)
            )
            positions = (
                torch.arange(len(members[0]), device=device) % n_features
                if n_features > 0
                else torch.empty(
                    len(members[0]), dtype=torch.long, device=device
                )
            )
            return ((base, rows, pattern_order[positions]),)

        columns = tuple(
            dict.fromkeys(column for schema in schemas for column in schema)
        )
        device = devices[0]
        size = len(columns)
        ranks = tuple(
            torch.randperm(size, generator=generator, device=device)
            for _ in range(3)
        )
        column_ids = {column: index for index, column in enumerate(columns)}

        states = []
        for schema, member_ids, schema_device in zip(
            schemas, members, devices, strict=True
        ):
            n_features = len(schema)
            if n_features == 0:
                empty = torch.empty(0, dtype=torch.long, device=schema_device)
                patterns = torch.empty(
                    len(member_ids), dtype=torch.long, device=schema_device
                )
                states.append((empty, empty, patterns))
                continue

            ids = torch.tensor(
                [column_ids[column] for column in schema],
                device=device,
            )
            present = torch.zeros(size, dtype=torch.bool, device=device)
            present[ids] = True
            local = torch.full((size,), -1, device=device)
            local[ids] = torch.arange(n_features, device=device)
            base, rows, pattern_order = (
                local[rank[present[rank]]].to(device=schema_device)
                for rank in ranks
            )
            positions = (
                torch.arange(len(member_ids), device=schema_device)
                % n_features
            )
            states.append((base, rows, pattern_order[positions]))
        return tuple(states)

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
        if len(self._locations) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._locations)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )

        tables: list[TableTensor] = []
        for member_id, (schema_id, position) in enumerate(self._locations):
            table = ensemble_table.table(member_id)
            fitted_host_permutation = self._host_permutations[schema_id][
                position
            ]
            expected_schema = self._schemas[schema_id]
            if inverse:
                expected_schema = tuple(
                    expected_schema[index] for index in fitted_host_permutation
                )
            if table.columns[Stype.numerical] != expected_schema:
                raise RuntimeError(
                    "ShuffleColumns must be transformed with the fitted "
                    "member schemas."
                )
            fitted_permutation = self._permutations[schema_id][position]
            permutation = (
                fitted_permutation.argsort() if inverse else fitted_permutation
            )
            if inverse:
                host_permutation = [0] * len(fitted_host_permutation)
                for destination, source in enumerate(fitted_host_permutation):
                    host_permutation[source] = destination
            else:
                host_permutation = fitted_host_permutation
            numerical = table.__class__(
                columns={
                    Stype.numerical: tuple(
                        table.columns[Stype.numerical][index]
                        for index in host_permutation
                    )
                },
                numerical=table.numerical.index_select(-1, permutation),
            )
            tables.append(
                cast(
                    TableTensor,
                    torch.cat(
                        (table.drop_stypes(Stype.numerical), numerical),
                        dim=-1,
                    ),
                )
            )

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def get_extra_state(
        self,
    ) -> tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[str, ...], ...],
    ]:
        r""":meta private:"""  # noqa: D415
        return self._host_permutations, self._locations, self._schemas

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._host_permutations, self._locations, self._schemas = cast(
            tuple[
                tuple[tuple[tuple[int, ...], ...], ...],
                tuple[tuple[int, int], ...],
                tuple[tuple[str, ...], ...],
            ],
            state,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
