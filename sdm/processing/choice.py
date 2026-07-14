from typing import cast

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Choice(Processor, InvertibleMixin):
    """Delegate to one option drawn uniformly at random.

    The option is drawn from the global CPU generator when the processor
    is fitted; seed with :func:`torch.manual_seed` to make it
    reproducible. Only the drawn option is fitted; refitting draws again.

    Args:
        args: Sequence of candidate processors.
    """

    supported_stypes = frozenset(Stype)

    def __init__(self, *args: Processor) -> None:
        super().__init__()
        self.options = torch.nn.ModuleList(args)
        self._index: int | None = None

    @property
    def selected(self) -> Processor:
        """The drawn option."""
        if self._index is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' has no drawn option; "
                "call 'fit()' before."
            )
        return cast(Processor, self.options[self._index])

    def _fit(self, table: TableTensor) -> None:
        self._index = int(torch.randint(len(self.options), (1,)).item())
        self.selected.fit(table)

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.selected.transform(table)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        fn = getattr(self.selected, "inverse_transform", None)
        if not callable(fn):
            raise AttributeError(
                f"'{self.selected.__class__.__name__}' object has no "
                f"attribute 'inverse_transform'"
            )
        return fn(table)

    def get_extra_state(self) -> int | None:  # noqa: D102
        return self._index

    def set_extra_state(self, state: int | None) -> None:  # noqa: D102
        if state is not None and not 0 <= state < len(self.options):
            raise ValueError(
                f"Cannot restore drawn option {state} on "
                f"'{self.__class__.__name__}' with {len(self.options)} "
                "options."
            )
        self._index = state

    def __repr__(self, *, indent: int = 0) -> str:
        inner = ",\n".join(
            cast(Processor, option).__repr__(indent=indent + 2)
            for option in self.options
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{inner},\n"
            f"{' ' * indent})"
        )
