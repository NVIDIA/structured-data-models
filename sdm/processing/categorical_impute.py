from typing import Literal

import torch

from sdm import CategoricalTensor, Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class CategoricalImpute(Processor):
    """Replace missing categorical values with fitted per-column values.

    Negative category codes are missing values. The fitted fill value is
    learned independently for every categorical column and applied without
    changing its category vocabulary.
    Transform inputs must use the fitted categorical column names, order, and
    category vocabularies. The processor raises if the encoded schema does not
    match.

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
        if strategy != "most_frequent":
            raise ValueError("strategy must be 'most_frequent'")
        self.strategy = strategy
        self._columns: tuple[str, ...] = ()
        self._categories: tuple[torch.Tensor, ...] = ()
        self.register_buffer(
            "_fill_values",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(self, input: TableTensor) -> None:
        data = input.categorical.as_tensor()
        self._check_codes(input)
        fill_values: list[torch.Tensor] = []
        columns = input.columns[Stype.categorical]
        for index, category in enumerate(input.categorical.categories):
            codes = data[..., index]
            observed = codes[codes >= 0].to(torch.long)
            if observed.numel() == 0:
                raise ValueError(
                    "Cannot fit 'CategoricalImpute' because categorical "
                    f"column '{columns[index]}' has no observed values."
                )

            counts = torch.bincount(
                observed,
                minlength=category.numel(),
            )
            fill_values.append(counts.argmax())

        self._fill_values = (
            torch.stack(fill_values)
            if len(fill_values) > 0
            else torch.empty(0, dtype=torch.long, device=data.device)
        )
        self._columns = columns
        self._categories = input.categorical.categories

    def _transform(self, input: TableTensor) -> TableTensor:
        self._check_schema(input)
        self._check_codes(input)
        data = input.categorical.as_tensor()
        data = torch.where(
            data < 0,
            self._fill_values.to(dtype=data.dtype),
            data,
        )
        categorical = CategoricalTensor(
            data=data,
            categories=input.categorical.categories,
        )
        return input.replace_blocks(categorical=categorical)

    def _check_schema(self, input: TableTensor) -> None:
        columns = input.columns[Stype.categorical]
        if columns != self._columns:
            raise ValueError(
                "Expected categorical columns to match the fitted names and "
                f"order (got {columns} and {self._columns})."
            )

        categories = input.categorical.categories
        for index, (actual, expected) in enumerate(
            zip(categories, self._categories)
        ):
            expected = expected.to(device=actual.device)
            if not torch.equal(actual, expected):
                raise ValueError(
                    "Expected the category vocabulary for categorical column "
                    f"'{columns[index]}' to match the fitted values and order."
                )

    @staticmethod
    def _check_codes(input: TableTensor) -> None:
        columns = input.columns[Stype.categorical]
        for index, category in enumerate(input.categorical.categories):
            codes = input.categorical.as_tensor()[..., index]
            if (codes >= category.numel()).any():
                raise ValueError(
                    f"Categorical column '{columns[index]}' contains a code "
                    "outside its category vocabulary."
                )
