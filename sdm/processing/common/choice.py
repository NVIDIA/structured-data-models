from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsembleProcessor,
    EnsembleTable,
    as_ensemble_processor,
)
from sdm.stype import Stype
from sdm.tensor import TableTensor


def _merge_outputs(
    *,
    selections: Sequence[int],
    positions: Mapping[int, Sequence[int]],
    outputs: Mapping[int, EnsembleTable],
) -> EnsembleTable:
    local_positions = {
        option: {position: local for local, position in enumerate(selected)}
        for option, selected in positions.items()
    }
    keys: dict[tuple[int, tuple[int, int]], int] = {}
    variants: list[TableTensor] = []
    member_to_variant: list[int] = []
    for member, option in enumerate(selections):
        result = outputs[option]
        location = result.member_to_variant[local_positions[option][member]]
        key = (option, location)
        if key not in keys:
            keys[key] = len(variants)
            group, variant = location
            variants.append(result.groups[group][variant])
        member_to_variant.append(keys[key])
    return EnsembleTable.pack(
        variants=variants,
        member_to_input_variant=member_to_variant,
    )


class Choice(EnsembleProcessor, InvertibleMixin):
    """Delegate to one selected option.

    The option is drawn when the processor is fitted; pass ``generator``
    to ``fit()`` to make it reproducible. The generator is also passed on
    to fit the drawn option. Only the drawn option is fitted; refitting
    draws again.

    Args:
        args: Sequence of candidate processors or stateless callables. Each
            callable accepts and returns a :class:`~sdm.tensor.TableTensor`.
        selection: ``"random"`` draws an option during fit. ``"round_robin"``
            assigns option ``member_id % len(options)`` in ensemble execution
            and selects the first option for the scalar API.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *args: object,
        selection: Literal["random", "round_robin"] = "random",
    ) -> None:
        super().__init__()
        if len(args) == 0:
            raise ValueError("'Choice' requires at least one option.")
        if selection not in {"random", "round_robin"}:
            raise ValueError("selection must be 'random' or 'round_robin'.")
        self.options = torch.nn.ModuleList(
            Processor.as_processor(arg) for arg in args
        )
        self.selection = selection
        self._index: int | None = None
        self._selections: tuple[int, ...] = ()
        self._positions: dict[int, tuple[int, ...]] = {}

    @property
    def selected(self) -> Processor:
        """The drawn option."""
        if self._index is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no drawn option; "
                "call 'fit()' before."
            )
        return cast(Processor, self.options[self._index])

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if self.selection == "round_robin":
            self._index = 0
        else:
            self._index = int(
                torch.randint(
                    len(self.options),
                    (1,),
                    generator=generator,
                    device=table.device,
                ).item()
            )
        self.selected.fit(table, generator=generator)

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.selected.transform(table)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        fn = getattr(self.selected, "inverse_transform", None)
        if not callable(fn):
            raise AttributeError(
                f"{self.selected.__class__.__name__!r} object has no "
                f"attribute 'inverse_transform'"
            )
        return fn(table)

    def _select_members(
        self,
        context: EnsembleFitContext,
    ) -> tuple[int, ...]:
        if self.selection == "round_robin":
            return tuple(
                member_id % len(self.options)
                for member_id in context.member_ids
            )
        return tuple(
            int(
                torch.randint(
                    len(self.options),
                    (1,),
                    generator=context.generator_for(member_id),
                ).item()
            )
            for member_id in context.member_ids
        )

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit each selected option once for its member subset."""
        for index, option in enumerate(tuple(self.options)):
            self.options[index] = as_ensemble_processor(
                cast(Processor, option)
            )

        self._selections = self._select_members(context)
        self._positions = {
            option: tuple(
                position
                for position, selected in enumerate(self._selections)
                if selected == option
            )
            for option in sorted(set(self._selections))
        }
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).fit_transform_ensemble(
                table._select_members(positions),
                context=context._select_members(positions).child(
                    f"option{option}"
                ),
            )
            for option, positions in self._positions.items()
        }
        return _merge_outputs(
            selections=self._selections,
            positions=self._positions,
            outputs=outputs,
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Reuse fitted member selections for transform."""
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).transform_ensemble(table._select_members(positions))
            for option, positions in self._positions.items()
        }
        return _merge_outputs(
            selections=self._selections,
            positions=self._positions,
            outputs=outputs,
        )

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Invert outputs through their selected options."""
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).inverse_transform_ensemble(table._select_members(positions))
            for option, positions in self._positions.items()
        }
        return _merge_outputs(
            selections=self._selections,
            positions=self._positions,
            outputs=outputs,
        )

    def get_extra_state(self) -> int | None:
        r""":meta private:"""  # noqa: D415
        return self._index

    def set_extra_state(self, state: int | None) -> None:
        r""":meta private:"""  # noqa: D415
        if state is not None and not 0 <= state < len(self.options):
            raise ValueError(
                f"Cannot restore drawn option {state} on "
                f"{self.__class__.__name__!r} with {len(self.options)} "
                "options."
            )
        self._index = state

    def __repr__(self, *, indent: int = 0) -> str:
        inner = ",\n".join(
            cast(Processor, option).__repr__(indent=indent + 2)
            for option in self.options
        )
        prefix = " " * indent
        if self.selection != "random":
            option_prefix = " " * (indent + 2)
            inner += f",\n{option_prefix}selection={self.selection!r}"
        return f"{prefix}{self.__class__.__name__}(\n{inner},\n{prefix})"
