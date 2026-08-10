from sdm import Stype, StypeLike, TableTensor
from sdm.processing.base import Processor


class DropStypes(Processor):
    r"""Remove all columns for specific semantic types.

    Args:
        stypes: Semantic column types to remove. All other semantic types are
            preserved unchanged.
    """

    requires_fit = False

    def __init__(self, *stypes: StypeLike) -> None:
        super().__init__()
        self._stypes = frozenset(Stype(stype) for stype in stypes)

    @property
    def operates_on_stypes(self) -> frozenset[Stype]:
        """Semantic types removed by this processor."""
        return self._stypes

    def _transform(self, table: TableTensor) -> TableTensor:
        """Drop configured semantic types from ``table``."""
        return table.drop_stypes(self._stypes)

    def __repr__(self, *, indent: int = 0) -> str:
        stypes = ", ".join(
            repr(stype.value) for stype in Stype if stype in self._stypes
        )
        return f"{' ' * indent}{self.__class__.__name__}({stypes})"
