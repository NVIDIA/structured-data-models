from typing import Literal, cast

import torch

from sdm import EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor


class RandomProjection(EnsembleProcessor):
    r"""Randomly project numerical columns.

    Args:
        channels: Number of output numerical columns.
        init: The projection weight initialization.
            ``"normal"`` samples Gaussian weights scaled by
            ``1 / sqrt(channels)``.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        channels: int,
        *,
        init: Literal["normal"] = "normal",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.init = init
        self._weights: BufferList[torch.Tensor] = BufferList()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        weights = []
        for i, group in enumerate(ensemble_table):
            num_members = ensemble_table.num_members_in_group(i)
            weight = group.numerical.new_empty(
                (
                    num_members,
                    *group.size()[1:-2],
                    self.channels,
                    group.numerical.size(-1),
                )
            )
            assert self.init == "normal"
            weight.normal_(std=self.channels**-0.5, generator=generator)
            weights.append(weight)
        self._weights = BufferList(weights)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        groups = []
        for i, group in enumerate(ensemble_table):
            positions = tuple(
                position
                for group_id, position in ensemble_table._locations
                if group_id == i
            )
            if (
                (
                    (
                        group.size(0) > 1
                        and positions != tuple(range(group.size(0)))
                    )
                    or (
                        group.size(0) == 1
                        and len(positions) > 1
                        and any(size > 1 for size in group.size()[1:-2])
                    )
                )
                and self._weights[i].shape[:-2]
                == (len(positions), *group.numerical.shape[1:-2])
                and not torch.is_autocast_enabled(group.device.type)
            ):
                numerical = group.numerical.new_empty(
                    (
                        len(positions),
                        *group.numerical.shape[1:-1],
                        self.channels,
                    )
                )
                # Write projections directly instead of gathering a full copy
                # of each input member or flattening expanded batch dimensions.
                for member, position in enumerate(positions):
                    torch.matmul(
                        group.numerical[position],
                        self._weights[i][member].transpose(-1, -2),
                        out=numerical[member],
                    )
                group = group.drop_stypes(Stype.numerical)
                if group.size(0) == 1:
                    group = cast(
                        TableTensor,
                        group.expand(len(positions), *group.size()[1:]),
                    )
                else:
                    index = torch.tensor(positions, device=group.device)
                    group = cast(TableTensor, group.index_select(0, index))
            else:
                group = ensemble_table.expanded_group(i)
                numerical = group.numerical @ self._weights[i].transpose(
                    -1, -2
                )
            projected = TableTensor(
                columns={
                    Stype.numerical: [f"rp_{i}" for i in range(self.channels)]
                },
                numerical=numerical,
            )
            group = cast(
                TableTensor,
                torch.cat(
                    [group.drop_stypes(Stype.numerical), projected], dim=-1
                ),
            )
            groups.append(group)

        locations = []
        group_position = [0] * ensemble_table.num_groups
        for group_id, _ in ensemble_table._locations:
            locations.append((group_id, group_position[group_id]))
            group_position[group_id] += 1

        return EnsembleTable(groups=groups, locations=locations)
