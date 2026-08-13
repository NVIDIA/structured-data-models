from typing import Literal, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable


class ShuffleCategories(EnsembleProcessor):
    """Independently permute the integer codes of categorical columns.

    One permutation per categorical column is drawn when the processor is
    fitted; pass ``generator`` to ``fit()`` to make the draws reproducible.
    For an ensemble, each repetition of the stored input tables receives one
    permutation. Tables with the same categorical schema share the permutation
    for the same repetition.
    Codes and their corresponding category vectors are permuted together so
    decoded values remain unchanged. Negative codes represent missing values
    and are preserved unchanged. Only categorical columns are supported; use
    :class:`~sdm.processing.StypeDispatch` to apply this processor to the
    categorical block of a mixed feature table.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            codes by a drawn offset, and ``"random"`` remaps the codes with
            a drawn permutation.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(
        self,
        method: Literal["shift", "random"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations: BufferList[BufferList[Tensor]] = BufferList()
        self._permutation_ids: tuple[int, ...] = ()

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._permutation_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._permutation_ids = cast(tuple[int, ...], state)

    def _draw_permutations(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> BufferList[Tensor]:
        device = table.categorical.device
        permutations: list[Tensor] = []
        orders: list[Tensor] = []
        cardinalities: list[int] = []
        offsets = [0]
        for category in table.categorical.categories:
            n_classes = category.numel()
            cardinalities.append(max(n_classes, 1))
            if self.method == "shift":
                offset = (
                    torch.zeros(
                        1,
                        dtype=table.categorical.code.dtype,
                        device=device,
                    )
                    if n_classes <= 1
                    else torch.randint(
                        n_classes,
                        (1,),
                        dtype=table.categorical.code.dtype,
                        generator=generator,
                        device=device,
                    )
                )
                permutation = offset
                order = (
                    torch.arange(n_classes, device=device) + offset
                ) % max(n_classes, 1)
            elif n_classes <= 1:
                permutation = torch.arange(
                    n_classes,
                    dtype=table.categorical.code.dtype,
                    device=device,
                )
                order = permutation
            else:
                assert self.method == "random"
                permutation = torch.randperm(
                    n_classes,
                    dtype=table.categorical.code.dtype,
                    generator=generator,
                    device=device,
                )
                order = permutation.argsort()
            permutations.append(permutation)
            orders.append(order)
            offsets.append(offsets[-1] + n_classes)

        permutation = (
            torch.cat(permutations)
            if len(permutations) > 0
            else torch.empty(
                0,
                dtype=table.categorical.code.dtype,
                device=device,
            )
        )
        order = (
            torch.cat(orders)
            if len(orders) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )
        return BufferList(
            (
                permutation,
                order,
                torch.tensor(
                    offsets,
                    dtype=table.categorical.code.dtype,
                    device=device,
                ),
                torch.tensor(
                    cardinalities,
                    dtype=table.categorical.code.dtype,
                    device=device,
                ),
            )
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        permutations_by_id: list[BufferList[Tensor]] = []
        permutation_ids: list[int] = []
        permutation_id_by_key: dict[
            tuple[int, torch.device, tuple[int, ...]], int
        ] = {}
        repetitions_by_location: dict[tuple[int, int], int] = {}

        for member_id in range(ensemble_table.num_members):
            location = ensemble_table._locations[member_id]
            repetition = repetitions_by_location.get(location, 0)
            repetitions_by_location[location] = repetition + 1
            table = ensemble_table.table(member_id)
            cardinalities = tuple(
                category.numel() for category in table.categorical.categories
            )
            key = (repetition, table.device, cardinalities)
            permutation_id = permutation_id_by_key.get(key)
            if permutation_id is None:
                permutation_id = len(permutations_by_id)
                permutation_id_by_key[key] = permutation_id
                permutations_by_id.append(
                    self._draw_permutations(table, generator=generator)
                )
            permutation_ids.append(permutation_id)

        self._permutations = BufferList(permutations_by_id)
        self._permutation_ids = tuple(permutation_ids)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutation_ids) != ensemble_table.num_members:
            raise RuntimeError(
                "ShuffleCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        positions_by_group_and_permutation: dict[
            tuple[int, int], list[int]
        ] = {}
        for location, permutation_id in zip(
            ensemble_table._locations,
            self._permutation_ids,
            strict=True,
        ):
            group_id, position = location
            positions = positions_by_group_and_permutation.setdefault(
                (group_id, permutation_id), []
            )
            if position not in positions:
                positions.append(position)

        permutation_ids_by_selection: dict[
            tuple[int, tuple[int, ...]], list[int]
        ] = {}
        for (
            group_id,
            permutation_id,
        ), positions in positions_by_group_and_permutation.items():
            permutation_ids_by_selection.setdefault(
                (group_id, tuple(positions)), []
            ).append(permutation_id)

        groups: list[TableTensor] = []
        output_locations: dict[tuple[int, int, int], tuple[int, int]] = {}
        for (
            group_id,
            positions,
        ), permutation_ids in permutation_ids_by_selection.items():
            group = ensemble_table._groups[group_id]
            if positions == tuple(range(group.size(0))):
                selected = group
            elif len(positions) == 1:
                selected = cast(
                    TableTensor,
                    group.narrow(0, positions[0], 1),
                )
            else:
                selected = cast(
                    TableTensor,
                    torch.stack([group[position] for position in positions]),
                )

            states = [
                self._permutations[permutation_id]
                for permutation_id in permutation_ids
            ]
            transformed = self._permute(selected, states)
            for permutation_id, output in zip(
                permutation_ids,
                transformed,
                strict=True,
            ):
                output_group_id = len(groups)
                groups.append(output)
                for output_position, input_position in enumerate(positions):
                    output_locations[
                        (group_id, input_position, permutation_id)
                    ] = (output_group_id, output_position)

        return EnsembleTable._from_groups(
            groups=groups,
            locations=tuple(
                output_locations[(*location, permutation_id)]
                for location, permutation_id in zip(
                    ensemble_table._locations,
                    self._permutation_ids,
                    strict=True,
                )
            ),
        )

    def _permute(
        self,
        table: TableTensor,
        states: list[BufferList[Tensor]],
    ) -> list[TableTensor]:
        code = table.categorical.code
        valid_mask = table.categorical.isfinite().unsqueeze(0)
        if self.method == "shift":
            permutations = torch.stack([state[0] for state in states])
            divisors = states[0][3]
            shape = (len(states), *(1,) * (code.dim() - 1), code.size(-1))
            mapped = (code.unsqueeze(0) - permutations.view(shape)) % divisors
        else:
            assert self.method == "random"
            permutations = torch.stack([state[0] for state in states])
            if permutations.size(1) == 0:
                mapped = code.unsqueeze(0).expand(len(states), *code.size())
            else:
                indices = code + states[0][2][:-1]
                mapped = permutations[:, indices.clamp_min(0).to(torch.long)]
        codes = torch.where(valid_mask, mapped, code.unsqueeze(0))

        outputs: list[TableTensor] = []
        for state, output_code in zip(states, codes, strict=True):
            categories: list[Tensor] = []
            offset = 0
            for category in table.categorical.categories:
                next_offset = offset + category.numel()
                categories.append(category[state[1][offset:next_offset]])
                offset = next_offset
            outputs.append(
                table.replace_blocks(
                    categorical=CategoricalTensor(
                        code=output_code,
                        categories=categories,
                    )
                )
            )
        return outputs

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
