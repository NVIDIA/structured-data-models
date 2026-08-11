from sdm import Stype, StypeLike, TableTensor
from sdm.processing.base import Processor


class DropStypes(Processor):
    r"""Remove all columns for specific semantic types.

    Args:
        stypes: Semantic column types to remove. All other semantic types are
            preserved unchanged.
    """

    requires_fit = False
    handled_stypes: frozenset[Stype]

    def __init__(self, *stypes: StypeLike) -> None:
        super().__init__()
        self._stypes = stypes
        self.handled_stypes = frozenset(Stype(stype) for stype in stypes)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.drop_stypes(self._stypes)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}{self._stypes!r}"
