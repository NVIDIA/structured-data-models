import torch
from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor


class Clip(Processor, InvertibleMixin):
    """Clamp feature columns to fitted quantile bounds.

    This transform is not reconstructive; ``inverse_transform`` intentionally
    returns its input unchanged.

    Args:
        q_low: Lower quantile in ``[0, 1]`` used as the per-column lower bound.
        q_high: Upper quantile in ``[0, 1]`` used as the per-column upper
            bound. Must satisfy ``0 <= q_low <= q_high <= 1``.
    """

    def __init__(
        self,
        *,
        q_low: float = 0.0,
        q_high: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0 <= q_low <= q_high <= 1:
            raise ValueError(
                "q_low and q_high must satisfy 0 <= q_low <= q_high <= 1."
            )
        self.q_low = q_low
        self.q_high = q_high
        self.register_buffer("lower_bound", torch.empty(0))
        self.register_buffer("upper_bound", torch.empty(0))

    def _fit(self, input: Tensor) -> None:
        quantiles = input.new_tensor([self.q_low, self.q_high])
        q_low, q_high = torch.quantile(input, quantiles, dim=0)
        self.lower_bound = q_low
        self.upper_bound = q_high

    def forward(self, input: Tensor) -> Tensor:
        """Clamp ``input`` to the fitted lower and upper bounds."""
        return input.clamp(min=self.lower_bound, max=self.upper_bound)

    def _inverse_transform(self, input: Tensor) -> Tensor:
        return input
