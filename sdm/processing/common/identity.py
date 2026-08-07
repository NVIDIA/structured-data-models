from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Identity(Processor, InvertibleMixin):
    r"""Return inputs unchanged."""

    operates_on_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table
