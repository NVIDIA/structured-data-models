from typing import Literal

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class ReduceEstimators(Processor):
    """Reduce the leading estimator dimension of model outputs.

    Input must be a numerical output table with shape ``[E, ..., R, O]``.
    ``E`` is the non-empty leading estimator dimension, ``R`` is the row
    dimension, and ``O`` is the output-column dimension. The result has shape
    ``[..., R, O]`` and retains the input column schema, device, and floating
    dtype.

    Place processors that support stacked outputs before this processor.
    Processors after it receive an already-reduced output table.

    Args:
        method: Reduction applied across estimators. Currently only
            ``"mean"`` is supported.
    """

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        method: Literal["mean"] = "mean",
    ) -> None:
        super().__init__()
        if method != "mean":
            raise ValueError("method must be 'mean'")
        self.method = method

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.dim() < 3:
            raise ValueError(
                "Expected a leading estimator dimension in an output table "
                f"with at least 3 dimensions (got {table.dim()}D)."
            )
        if table.size(0) == 0:
            raise ValueError("Expected at least one estimator.")

        numerical = _as_float(table.numerical).mean(dim=0)
        return table.__class__(
            columns={Stype.numerical.value: table.columns[Stype.numerical]},
            numerical=numerical,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
