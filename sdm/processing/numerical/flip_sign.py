import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor


class FlipSign(Processor, InvertibleMixin):
    """Randomly negate numerical columns.

    Args:
        probability: Probability of negating a numerical column.
        quantile_output: Whether inverse transformation operates on ordered
            quantile predictions. Quantiles are reversed for negated targets.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        probability: float = 0.5,
        *,
        quantile_output: bool = False,
    ) -> None:
        super().__init__()
        self.probability = probability
        self.quantile_output = quantile_output
        self.register_buffer("sign", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        shape = (*table.numerical.size()[:-2], 1, table.numerical.size(-1))
        self.sign = table.numerical.new_empty(shape)
        self.sign.bernoulli_(self.probability, generator=generator)
        self.sign.mul_(-2).add_(1)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical * self.sign)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        output = self._transform(table)
        if not self.quantile_output:
            return output
        return output.replace_blocks(
            numerical=output.numerical.where(
                self.sign > 0,
                output.numerical.flip(-1),
            )
        )
