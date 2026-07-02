from typing import Any

from sdm.processing.base import InvertibleMixin, Processor


class Identity(Processor, InvertibleMixin):
    """Return inputs unchanged.

    This stateless processor is useful as an explicit no-op in recipe phases.
    """

    requires_fit = False

    def forward(self, input: Any) -> Any:
        """Return ``input`` unchanged."""
        return input

    def _inverse_transform(self, input: Any) -> Any:
        return input
