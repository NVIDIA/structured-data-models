from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Identity(Processor, InvertibleMixin):
    """Return inputs unchanged.

    This stateless processor is useful as an explicit no-op in recipe phases.
    """

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, input: TableTensor) -> TableTensor:
        """Return ``input`` unchanged."""
        return input

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return input
