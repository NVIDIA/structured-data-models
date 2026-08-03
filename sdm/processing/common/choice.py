from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.ensemble import (
    EnsembleProcessor,
    EnsembleProcessorAdapter,
)
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


def _merge_ensemble_outputs(
    selections: Sequence[int],
    outputs: Mapping[int, EnsembleTable],
) -> EnsembleTable:
    representations: list[TableTensor] = []
    locations: dict[tuple[int, tuple[int, int]], int] = {}
    member_representation_ids = []
    for member_id, option in enumerate(selections):
        output = outputs[option]
        location = output.member_location(member_id)
        key = (option, location)
        if key not in locations:
            locations[key] = len(representations)
            representations.append(output.representation(member_id))
        member_representation_ids.append(locations[key])

    return EnsembleTable.from_representations(
        representations=representations,
        member_representation_ids=member_representation_ids,
    )


class Choice(EnsembleProcessor, InvertibleMixin):
    """Delegate each table or ensemble member to one selected option.

    The option is drawn when the processor is fitted; pass ``generator``
    to ``fit()`` to make it reproducible. The generator is also passed on
    to fit the drawn option. Only the drawn option is fitted; refitting
    draws again. Ensemble execution selects one option per member and evaluates
    every selected option over the packed table representations.

    Args:
        args: Sequence of candidate processors or stateless callables. Each
            callable accepts and returns a :class:`~sdm.tensor.TableTensor`.
        selection: Selection strategy. Round-robin assigns options by ensemble
            member position and selects the first option for a single table.
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
            raise ValueError("Expected 'random' or 'round_robin' selection.")
        self.options = torch.nn.ModuleList(
            Processor.as_processor(arg) for arg in args
        )
        self.selection = selection
        self._index: int | None = None
        self._selections: tuple[int, ...] = ()

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
            device = table.device if generator is None else generator.device
            self._index = int(
                torch.randint(
                    len(self.options),
                    (1,),
                    generator=generator,
                    device=device,
                ).item()
            )
        self.selected.fit(table, generator=generator)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._fit(table, generator=generator)
        return self.selected.transform(table)

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

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        for index, option in enumerate(tuple(self.options)):
            self.options[index] = EnsembleProcessorAdapter.adapt(
                cast(Processor, option)
            )

        if self.selection == "round_robin":
            self._selections = tuple(
                member_id % len(self.options)
                for member_id in range(table.num_members)
            )
        else:
            device = (
                next(table.iter_packed_representations()).device
                if generator is None
                else generator.device
            )
            self._selections = tuple(
                torch.randint(
                    len(self.options),
                    (table.num_members,),
                    generator=generator,
                    device=device,
                ).tolist()
            )

        # Each option sees stable member positions, including nested choices.
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).fit_transform_ensemble(
                table,
                generator=generator,
            )
            for option in sorted(set(self._selections))
        }
        return _merge_ensemble_outputs(self._selections, outputs)

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        if len(self._selections) != table.num_members:
            raise RuntimeError(
                "Choice must be fitted with the same number of ensemble "
                "members before transform."
            )
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).transform_ensemble(table)
            for option in sorted(set(self._selections))
        }
        return _merge_ensemble_outputs(self._selections, outputs)

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        """Invert members through their selected options."""
        self._check_is_fitted()
        outputs = {}
        for option in sorted(set(self._selections)):
            processor = cast(EnsembleProcessor, self.options[option])
            inverse = getattr(processor, "inverse_transform_ensemble", None)
            if not callable(inverse):
                raise AttributeError(
                    f"{processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform_ensemble'"
                )
            outputs[option] = inverse(table)
        return _merge_ensemble_outputs(self._selections, outputs)

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
        if self.selection != "random":
            inner += f",\n{' ' * (indent + 2)}selection={self.selection!r}"
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{inner},\n"
            f"{' ' * indent})"
        )
