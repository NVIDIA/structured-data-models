from collections.abc import Sequence

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Choice(Processor, InvertibleMixin):
    """Select one option uniformly at random and delegate to it.

    The selection happens once, at construction, and consumes a single
    draw from the global CPU generator; seed with
    :func:`torch.manual_seed` to make the selection reproducible. Only
    the selected option is kept and registered as a submodule;
    unselected options are discarded, so they are never fitted, moved
    between devices, or checkpointed.

    Constructing the same recipe repeatedly therefore yields ensemble
    members with independently selected processing routes.

    Args:
        options: Non-empty sequence of candidate processors.
    """

    supported_stypes = frozenset(Stype)

    def __init__(self, options: Sequence[Processor]) -> None:
        super().__init__()
        if len(options) == 0:
            raise ValueError("options must be non-empty.")
        index = int(torch.randint(len(options), (1,)).item())
        self.selected: Processor = options[index]
        self.requires_fit = self.selected.requires_fit

    def _fit(self, input: TableTensor) -> None:
        self.selected.fit(input)

    def _transform(self, input: TableTensor) -> TableTensor:
        return self.selected.transform(input)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        fn = getattr(self.selected, "inverse_transform", None)
        if not callable(fn):
            raise AttributeError(
                f"'{self.selected.__class__.__name__}' object has no "
                f"attribute 'inverse_transform'"
            )
        return fn(input)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{self.selected.__repr__(indent=indent + 2)},\n"
            f"{' ' * indent})"
        )
