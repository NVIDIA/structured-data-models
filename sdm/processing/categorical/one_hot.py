from typing import cast

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class OneHot(Processor):
    r"""Convert categorical columns to one-hot numerical columns.

    Negative and out-of-vocabulary codes map to all-zero vectors. Apply
    :class:`~sdm.processing.AlignCategories` first when context and query
    tables may use different category vocabularies.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        counts = tuple(
            len(category) for category in table.categorical.categories
        )
        total_count = sum(counts)
        if total_count == 0:
            encoded = table.numerical.new_empty((*table.size()[:-1], 0))
        else:
            count = torch.tensor(counts, device=table.device)
            offset = count.cumsum(0) - count
            code = table.categorical.code.to(torch.long)
            valid = (code >= 0) & (code < count)
            index = (code + offset).clamp(min=0, max=total_count - 1)
            encoded = table.numerical.new_zeros(
                (*code.size()[:-1], total_count)
            )
            encoded.scatter_add_(
                -1,
                index,
                valid.to(encoded.dtype),
            )

        columns = tuple(
            f"{column}__{category_id}"
            for column, num_categories in zip(
                table.columns[Stype.categorical], counts, strict=True
            )
            for category_id in range(num_categories)
        )
        out = TableTensor(
            columns={Stype.numerical: columns},
            numerical=encoded,
        )
        return cast(
            TableTensor,
            torch.cat((table.drop_stypes(Stype.categorical), out), dim=-1),
        )
