# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator, Sequence
from typing import Literal, cast

import torch
from torch.nn import ModuleList

from sdm import EnsembleTable, TableLike, TableTensor
from sdm.processing import InvertibleMixin, Processor


class Choice(Processor, InvertibleMixin):
    """Route each table or ensemble member through one selected option.

    Options are selected when the processor is fitted. Only selected options
    are fitted, and refitting selects again.

    Args:
        args: Candidate processors or stateless callables.
        method: How to select options. ``"random"`` samples uniformly;
            ``"round_robin"`` assigns options by ensemble member position and
            selects the first option for a single table.
    """

    requires_fit = True

    def __init__(
        self,
        *args: object,
        method: Literal["random", "round_robin"] = "random",
    ) -> None:
        super().__init__()
        options = []
        for arg in args:
            options.append(Processor.as_processor(arg))
        self.options: ModuleList[Processor] = ModuleList(options)
        self.handles_stypes = self._merge_stypes(self.options)
        self.method = method
        self._option_ids: tuple[int, ...] = ()

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._option_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._option_ids = cast(tuple[int, ...], state)

    def _draw_option_ids(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[int, ...]:
        num_members = len(table) if isinstance(table, EnsembleTable) else 1
        if self.method == "round_robin":
            return tuple(
                member_id % len(self.options)
                for member_id in range(num_members)
            )

        assert self.method == "random"
        device = table.device if generator is None else generator.device
        return tuple(
            torch.randint(
                len(self.options),
                (num_members,),
                generator=generator,
                device=device,
            ).tolist()
        )

    def _tables_by_option(
        self,
        table: TableLike,
    ) -> Iterator[tuple[int, TableLike]]:
        if isinstance(table, TableTensor):
            yield self._option_ids[0], table
            return
        member_ids_by_option: dict[int, list[int]] = {}
        for member_id, option_id in enumerate(self._option_ids):
            member_ids_by_option.setdefault(option_id, []).append(member_id)
        # Select lazily, since selecting members may copy them.
        for option_id, member_ids in sorted(member_ids_by_option.items()):
            yield option_id, table[member_ids]

    def _gather_outputs(
        self,
        table: TableLike,
        outputs: dict[int, TableLike],
    ) -> TableLike:
        if isinstance(table, TableTensor):
            return outputs[self._option_ids[0]]
        tables = []
        member_ids = []
        next_member_id_by_option: dict[int, int] = {}
        for option_id in self._option_ids:
            tables.append(outputs[option_id])
            member_id = next_member_id_by_option.get(option_id, 0)
            member_ids.append(member_id)
            next_member_id_by_option[option_id] = member_id + 1
        return table.gather_members(
            tables=cast(Sequence[EnsembleTable], tables),
            member_ids=member_ids,
        )

    def _check_num_members(self, table: TableLike) -> None:
        num_members = len(table) if isinstance(table, EnsembleTable) else 1
        if len(self._option_ids) != num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._option_ids)} ensemble members, but got "
                f"{num_members}."
            )

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._option_ids = self._draw_option_ids(
            table,
            generator=generator,
        )
        for option_id, selected_table in self._tables_by_option(table):
            self.options[option_id].fit(selected_table, generator=generator)

    def _fit_transform(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> TableLike:
        self._option_ids = self._draw_option_ids(
            table,
            generator=generator,
        )
        outputs = {
            option_id: self.options[option_id].fit_transform(
                selected_table,
                generator=generator,
            )
            for option_id, selected_table in self._tables_by_option(table)
        }
        return self._gather_outputs(table, outputs)

    def _transform(
        self,
        table: TableLike,
    ) -> TableLike:
        self._check_num_members(table)
        outputs = {
            option_id: self.options[option_id].transform(selected_table)
            for option_id, selected_table in self._tables_by_option(table)
        }
        return self._gather_outputs(table, outputs)

    def _inverse_transform(
        self,
        table: TableLike,
    ) -> TableLike:
        self._check_num_members(table)
        outputs = {}
        for option_id, selected_table in self._tables_by_option(table):
            processor = self.options[option_id]
            outputs[option_id] = processor.inverse_transform(selected_table)
        return self._gather_outputs(table, outputs)

    def __repr__(self, *, indent: int = 0) -> str:
        inner = ",\n".join(
            option.__repr__(indent=indent + 2) for option in self.options
        )
        if self.method != "random":
            inner += f",\n{' ' * (indent + 2)}method={self.method!r}"
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{inner},\n"
            f"{' ' * indent})"
        )
