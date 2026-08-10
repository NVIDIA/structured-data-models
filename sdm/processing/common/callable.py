from collections.abc import Callable

from sdm import Stype, TableTensor
from sdm.processing import Processor


class Callable(Processor):
    """Adapt a stateless callable to the :class:`Processor` interface."""

    operates_on_stypes = frozenset(Stype)
    unoperated_stype_policy = "opaque"
    requires_fit = False

    def __init__(self, function: Callable[[TableTensor], TableTensor]) -> None:
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
        return f"{' ' * indent}{self.__class__.__name__}({name})"
