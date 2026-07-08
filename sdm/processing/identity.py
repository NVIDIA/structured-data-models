from sdm.processing import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Identity(Processor, InvertibleMixin):
    r"""Return inputs unchanged."""

    #:
    supported_stypes = frozenset(Stype) - {Stype.id}
    #:
    requires_fit = False

    def _transform(self, input: TableTensor) -> TableTensor:
        return input

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return input
