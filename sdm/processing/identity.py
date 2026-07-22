from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Identity(Processor, InvertibleMixin):
    """Return inputs unchanged.

    This stateless processor is useful as an explicit no-op in recipe phases.
    It preserves every dimension, so output tables may be either stacked or
    already reduced across estimators.
    """

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``table`` unchanged."""
        return table

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table
