from typing import cast

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class PairwiseInteractions(Processor):
    """Append pairwise products and differences of numerical columns.

    Applied only when the block has at most ``max_columns`` columns; wider
    blocks pass through unchanged.

    Args:
        max_columns: Largest column count that receives interactions.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, max_columns: int = 12) -> None:
        super().__init__()
        self.max_columns = max_columns

    def _transform(self, table: TableTensor) -> TableTensor:
        x = table.numerical
        names = list(table.columns[Stype.numerical])
        d = x.size(-1)
        if not 2 <= d <= self.max_columns:
            return table
        i, j = torch.triu_indices(d, d, offset=1, device=x.device)
        products = x[..., i] * x[..., j]
        differences = x[..., i] - x[..., j]
        names += [
            f"{names[a]}*{names[b]}" for a, b in zip(i.tolist(), j.tolist())
        ]
        names += [
            f"{names[a]}-{names[b]}" for a, b in zip(i.tolist(), j.tolist())
        ]
        out = TableTensor(
            columns={Stype.numerical: names},
            numerical=torch.cat([x, products, differences], dim=-1),
        )
        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.numerical), out], dim=-1),
        )
