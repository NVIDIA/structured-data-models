from typing import Literal

import torch

from sdm import CategoricalTensor, Stype
from sdm.processing.base import Processor
from sdm.processing.categorical._categorical import _check_categorical_codes
from sdm.tensor import TableTensor


class CategoricalImpute(Processor):
    """Replace missing categorical values with fitted per-column values.

    Negative category codes are missing values. The fitted fill value is
    learned independently for every categorical column and applied without
    changing its category vocabulary.
    Transform inputs must use the fitted per-column category vocabularies.
    The processor raises if they do not match. Column names are not
    validated. Use :class:`~sdm.processing.CategoricalAlign` before this
    processor when training and transform inputs were tensorized
    independently.

    Args:
        strategy: Imputation strategy. ``"most_frequent"`` selects the most
            common observed category in each fitted column. Ties select the
            lowest category code.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(
        self,
        *,
        strategy: Literal["most_frequent"] = "most_frequent",
    ) -> None:
        super().__init__()
        # TODO: Add a strategy that encodes missing values as their own
        # category instead of imputing an observed one.
        if strategy != "most_frequent":
            raise ValueError("strategy must be 'most_frequent'")
        self.strategy = strategy
        self._categories: tuple[torch.Tensor, ...] = ()
        self.register_buffer(
            "_fill_values",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        data = table.categorical
        _check_categorical_codes(table)
        fill_values: list[torch.Tensor] = []
        columns = table.columns[Stype.categorical]
        for index, category in enumerate(table.categorical.categories):
            codes = data[..., index]
            observed = codes[codes >= 0].to(torch.long)
            if observed.numel() == 0:
                raise ValueError(
                    "Cannot fit 'CategoricalImpute' because categorical "
                    f"column '{columns[index]}' has no observed values."
                )

            counts = observed.bincount(minlength=category.numel())
            fill_values.append(counts.argmax())

        self._fill_values = (
            torch.stack(fill_values)
            if len(fill_values) > 0
            else torch.empty(0, dtype=torch.long, device=data.device)
        )
        self._categories = table.categorical.categories

    def _transform(self, table: TableTensor) -> TableTensor:
        self._check_categories(table)
        _check_categorical_codes(table)
        data = table.categorical.where(
            table.categorical >= 0,
            self._fill_values.to(dtype=table.categorical.dtype),
        )
        categorical = CategoricalTensor(
            data=data,
            categories=table.categorical.categories,
        )
        return table.replace_blocks(categorical=categorical)

    def _check_categories(self, table: TableTensor) -> None:
        columns = table.columns[Stype.categorical]
        if len(table.categorical.categories) != len(self._categories):
            raise ValueError(
                f"Expected {len(self._categories)} fitted categorical "
                f"columns (got {len(columns)})."
            )
        for index, (actual, expected) in enumerate(
            zip(table.categorical.categories, self._categories)
        ):
            expected = expected.to(device=actual.device)
            if not actual.equal(expected):
                raise ValueError(
                    "Expected the category vocabulary for categorical column "
                    f"'{columns[index]}' to match the fitted values and "
                    "order. "
                    "Use 'CategoricalAlign' before this processor for "
                    "independently tensorized inputs."
                )
