import torch

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class ClipQuantiles(Processor):
    """Clamp feature columns to fitted quantile bounds.

    Values outside the fitted bounds are discarded, so this processor is not
    invertible. Quantile transform bounds are fitted independently for each
    feature column.

    Args:
        q_low: Lower quantile in ``[0, 1]`` used as the per-column lower bound.
        q_high: Upper quantile in ``[0, 1]`` used as the per-column upper
            bound. Must satisfy ``0 <= q_low <= q_high <= 1``.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        q_low: float = 0.0,
        q_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.q_low = q_low
        self.q_high = q_high
        self.register_buffer("min_value", torch.empty(0))
        self.register_buffer("max_value", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.min_value, self.max_value = torch.quantile(
            table.numerical,
            q=table.numerical.new_tensor([self.q_low, self.q_high]),
            dim=-2,
            keepdim=True,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical.clamp(self.min_value, self.max_value)
        return table.replace_blocks(numerical=numerical)

    # TODO REPR
