from collections.abc import Callable
from typing import TypeAlias

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor

ProcessorCallable: TypeAlias = Callable[[TableTensor], TableTensor]
ProcessorLike: TypeAlias = Processor | ProcessorCallable


class _CallableProcessor(Processor):
    """Adapt a stateless callable to the :class:`Processor` interface."""

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(self, function: ProcessorCallable) -> None:
        super().__init__()
        self.function = function

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.function(table)

    def __repr__(self, *, indent: int = 0) -> str:
        name = getattr(
            self.function,
            "__name__",
            self.function.__class__.__name__,
        )
        if name == "<lambda>":
            name = "lambda"
        return f"{' ' * indent}{name}"


def as_processor(step: ProcessorLike, *, label: str) -> Processor:
    """Normalize a processor or stateless callable to a processor."""
    if isinstance(step, Processor):
        return step
    if callable(step):
        return _CallableProcessor(step)
    raise TypeError(
        f"{label} must be a Processor or callable, got "
        f"{step.__class__.__name__}."
    )
