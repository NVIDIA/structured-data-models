from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor


class Identity(Processor, InvertibleMixin):
    """Return inputs unchanged.

    This stateless processor is useful as an explicit no-op in recipe phases.
    """

    requires_fit = False

    def _transform(self, input: Tensor) -> Tensor:
        """Return ``input`` unchanged."""
        return input

    def _inverse_transform(self, input: Tensor) -> Tensor:
        return input
