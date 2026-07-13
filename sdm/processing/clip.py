import torch

from sdm.processing._utils import _as_float
from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Clip(Processor, InvertibleMixin):
    """Clamp feature columns to fitted quantile bounds.

    This transform is not reconstructive; ``inverse_transform`` intentionally
    returns its input unchanged. Quantile bounds are fitted independently for
    each feature column.

    Args:
        q_low: Lower quantile in ``[0, 1]`` used as the per-column lower bound.
        q_high: Upper quantile in ``[0, 1]`` used as the per-column upper
            bound. Must satisfy ``0 <= q_low <= q_high <= 1``.
    """

    supported_stypes = frozenset({Stype.numerical})

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

    def _fit(self, inp: TableTensor) -> None:
        numerical = _as_float(inp.numerical)
        quantiles = numerical.new_tensor([self.q_low, self.q_high])
        q_low, q_high = torch.quantile(numerical, quantiles, dim=0)
        self.lower_bound = q_low
        self.upper_bound = q_high

    def _transform(self, inp: TableTensor) -> TableTensor:
        """Clamp ``inp`` to the fitted lower and upper bounds."""
        numerical = _as_float(inp.numerical).clamp(
            min=self.lower_bound,
            max=self.upper_bound,
        )
        return inp.replace_blocks(numerical=numerical)

    def _inverse_transform(self, inp: TableTensor) -> TableTensor:
        return inp
