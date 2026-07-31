import torch

from sdm import CategoricalTensor, Stype
from sdm.processing.base import Processor
from sdm.processing.categorical._categorical import _check_categorical_codes
from sdm.tensor import TableTensor


class ImputeMode(Processor):
    """Replace missing categorical values with fitted per-column modes.

    Negative category codes are missing values. The fitted fill value is
    the most frequent observed category code learned independently for every
    categorical column and applied without changing its category vocabulary.
    Ties select the lowest category code.

    Transform inputs must use the fitted per-column category vocabularies.
    The processor raises if they do not match. Column names are not
    validated. Use :class:`~sdm.processing.AlignCategories` before this
    processor when training and transform inputs were tensorized
    independently.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(self) -> None:
        super().__init__()
        # TODO: Add a separate processor that encodes missing values as their
        # own category instead of imputing an observed one.
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
            codes = data[..., index]  # [*batch, n_samples]
            observed = codes >= 0
            if not bool(observed.any(dim=-1).all()):
                raise ValueError(
                    "Cannot fit 'ImputeMode' because categorical "
                    f"column {columns[index]!r} has no observed values."
                )

            counts = torch.zeros(
                (*codes.shape[:-1], category.numel()),
                dtype=torch.long,
                device=codes.device,
            )  # [*batch, n_categories]
            counts.scatter_add_(
                -1,
                codes.clamp_min(0).to(torch.long),
                observed.to(torch.long),
            )  # [*batch, n_categories]
            fill_values.append(counts.argmax(dim=-1))

        self._fill_values = (
            torch.stack(fill_values, dim=-1)
            if len(fill_values) > 0
            else torch.empty(0, dtype=torch.long, device=data.device)
        )
        if data.dim() > 2:
            self._fill_values = self._fill_values.unsqueeze(-2)
        self._categories = table.categorical.categories

    def _transform(self, table: TableTensor) -> TableTensor:
        self._check_categories(table)
        _check_categorical_codes(table)
        code = table.categorical.where(
            table.categorical >= 0,
            self._fill_values.to(dtype=table.categorical.dtype),
        )
        categorical = CategoricalTensor(
            code=code,
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
                    f"{columns[index]!r} to match the fitted values and "
                    "order. "
                    "Use 'AlignCategories' before this processor for "
                    "independently tensorized inputs."
                )
