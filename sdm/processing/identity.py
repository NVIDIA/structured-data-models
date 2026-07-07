from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class Identity(Processor, InvertibleMixin):
    """Return inputs unchanged.

    This stateless processor is useful as an explicit no-op in recipe phases.
    """

    requires_fit = False

    def forward(self, input: TableTensor) -> TableTensor:
        """Return ``input`` unchanged."""
        return input

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        return input
