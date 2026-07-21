import torch

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class PCA(Processor):
    """Project numerical columns onto their principal components.

    The mean and components are fitted on the context table via a singular
    value decomposition of the centered data. The effective dimension is
    capped at ``min(dim, num_rows, num_features)``. Output columns are named
    ``pca_0, ..., pca_{d-1}``.

    Args:
        dim: Number of principal components to keep.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(self, *, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"'dim' must be positive (got {dim})")
        self.dim = dim
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = _as_float(table.numerical)  # [N, F]
        self.mean = numerical.mean(dim=0)
        centered = numerical - self.mean
        # Economy SVD; right-singular vectors are the principal axes.
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        dim = min(self.dim, vh.size(0))
        self.components = vh[:dim].T  # [F, dim]

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = _as_float(table.numerical) - self.mean
        projected = numerical @ self.components  # [N, dim]
        return table.__class__(
            columns={
                Stype.numerical: tuple(
                    f"pca_{i}" for i in range(projected.size(-1))
                ),
            },
            numerical=projected,
        )
