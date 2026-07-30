from typing import Literal, cast

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Choice(Processor, InvertibleMixin):
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
    member_specific_fit = True

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
