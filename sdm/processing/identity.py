from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Identity(Processor, InvertibleMixin):
    """Return inputs unchanged.

    This stateless processor is useful as an explicit no-op in recipe phases.
    """

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, inp: TableTensor) -> TableTensor:
        """Return ``inp`` unchanged."""
        return inp

    def _inverse_transform(self, inp: TableTensor) -> TableTensor:
        return inp
