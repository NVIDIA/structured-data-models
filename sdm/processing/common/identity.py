from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class Identity(Processor, InvertibleMixin):
    r"""Return inputs unchanged."""

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table
