import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor, Stype
from sdm.processing._categorical import _check_categorical_codes
from sdm.processing.base import Processor
from sdm.tensor import TableTensor

_HOST_MAPPED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class CategoricalAlign(Processor):
    """Align categorical codes to vocabularies fitted on training rows.

    Categorical codes are the integer indices into a column's category
    vocabulary stored by :class:`~sdm.CategoricalTensor`, following
    :attr:`pandas.Categorical.codes` semantics; negative codes encode missing
    values. Fitting stores the category values that are actually observed in
    each categorical column. Transforming remaps input codes by category value
    to those fitted vocabularies. Missing values and categories not observed
    during fitting are encoded as ``-1``.

    This allows independently tensorized training and query tables to share a
    categorical schema. It also removes categories that occur only outside a
    sliced training context from a jointly inferred vocabulary. Only
    categorical columns are supported; use
    :class:`~sdm.processing.StypeDispatch` for mixed feature tables.

    String and unsigned integer vocabularies are matched through host metadata
    because their required tensor operations are unavailable on every device.
    This path performs linear Python work in the vocabulary size; a vectorized
    implementation may be preferable if these category types need to scale.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(self) -> None:
        super().__init__()
        self._categories: tuple[Tensor, ...] = ()

    def _fit(self, table: TableTensor) -> None:
        _check_categorical_codes(table)
        data = table.categorical
        columns = table.columns[Stype.categorical]
        categories: list[Tensor] = []
        for index, category in enumerate(table.categorical.categories):
            if category.is_complex():
                raise ValueError(
                    "CategoricalAlign does not support complex category "
                    f"values for categorical column '{columns[index]}'."
                )
            codes = data[..., index].reshape(-1)  # [num_rows]
            positions = torch.arange(
                end=codes.numel(),
                device=data.device,
            )  # [num_rows]
            # [num_local_categories]
            first_positions = torch.full(
                size=(category.numel(),),
                fill_value=codes.numel(),
                dtype=torch.long,
                device=data.device,
            )
            observed = codes >= 0
            first_positions.scatter_reduce_(
                dim=0,
                index=codes[observed].to(torch.long),
                src=positions[observed],
                reduce="amin",
                include_self=True,
            )
            # [num_observed_categories]
            observed = (first_positions < codes.numel()).nonzero().view(-1)
            observed = observed[first_positions[observed].argsort()]
            categories.append(self._select_categories(category, observed))

        self._categories = tuple(categories)

    def _transform(self, table: TableTensor) -> TableTensor:
        _check_categorical_codes(table)

        columns = table.columns[Stype.categorical]
        # Start from all-missing output codes; the per-column loop below only
        # overwrites observed positions, so missing and unseen values stay -1.
        data = torch.full_like(table.categorical, -1)
        for index, (actual, expected) in enumerate(
            zip(table.categorical.categories, self._categories, strict=True)
        ):
            codes = table.categorical[..., index]
            observed = codes >= 0
            if not observed.any():
                continue

            mapping = self._category_mapping(
                actual=actual,
                expected=expected,
                device=data.device,
                column=columns[index],
            )
            data[..., index][observed] = mapping[
                codes[observed].to(torch.long)
            ].to(data.dtype)

        categorical = CategoricalTensor(
            data=data,
            categories=tuple(
                category.to(device=data.device)
                for category in self._categories
            ),
        )
        return table.replace_blocks(categorical=categorical)

    @staticmethod
    def _select_categories(category: Tensor, index: Tensor) -> Tensor:
        if category.dtype in _HOST_MAPPED_DTYPES:
            values = category.tolist()
            return torch.tensor(
                data=[values[i] for i in index.tolist()],
                dtype=category.dtype,
                device=category.device,
            )
        return category.index_select(0, index.to(device=category.device))

    @staticmethod
    def _category_mapping(
        actual: Tensor,
        expected: Tensor,
        device: torch.device,
        column: str,
    ) -> Tensor:
        if expected.numel() == 0:
            # An all-missing fit stores an empty vocabulary whose dtype is a
            # placeholder, so value-type compatibility cannot be validated;
            # every query value is unseen and maps to -1.
            return torch.full(
                size=(actual.numel(),),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )

        if isinstance(actual, StringTensor) != isinstance(
            expected, StringTensor
        ):
            raise ValueError(
                "Expected category value types to match the fitted values for "
                f"categorical column '{column}'."
            )

        if isinstance(actual, StringTensor):
            # String columns store their category vocabulary as StringTensor.
            # StringTensor has no element-wise equality operation. Category
            # vocabularies are metadata, so only their values move to the host;
            # row-wise codes remain on their original device.
            expected_index = {
                value: index for index, value in enumerate(expected.tolist())
            }
            return torch.tensor(
                data=[
                    expected_index.get(value, -1) for value in actual.tolist()
                ],
                dtype=torch.long,
                device=device,
            )

        actual = actual.to(device=device)
        expected = expected.to(device=device)
        if actual.dtype != expected.dtype:
            raise ValueError(
                "Expected category value dtypes to match the fitted "
                f"values for categorical column '{column}' "
                f"(got {actual.dtype} and "
                f"{expected.dtype})."
            )
        if actual.dtype in _HOST_MAPPED_DTYPES:
            expected_index = {
                value: index for index, value in enumerate(expected.tolist())
            }
            return torch.tensor(
                data=[
                    expected_index.get(value, -1) for value in actual.tolist()
                ],
                dtype=torch.long,
                device=device,
            )

        # [num_actual_categories]
        mapping = torch.full(
            size=(actual.numel(),),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        dtype = actual.dtype
        if dtype == torch.bool:
            dtype = torch.uint8
        actual = actual.to(dtype=dtype)
        expected = expected.to(dtype=dtype)

        if dtype.is_floating_point:
            # [num_actual_categories]
            actual_nan = actual.isnan()
            # [num_fitted_categories]
            expected_nan = expected.isnan()
            if actual_nan.any() and expected_nan.any():
                mapping[actual_nan] = expected_nan.to(torch.int64).argmax()
        else:
            actual_nan = torch.zeros_like(actual, dtype=torch.bool)
            expected_nan = torch.zeros_like(expected, dtype=torch.bool)

        # [num_actual_non_nan] and [num_fitted_non_nan]
        actual_indices = (~actual_nan).nonzero().view(-1)
        expected_indices = (~expected_nan).nonzero().view(-1)
        if actual_indices.numel() == 0 or expected_indices.numel() == 0:
            return mapping

        # [num_fitted_non_nan]
        expected_values, permutation = expected[expected_indices].sort()
        actual_values = actual[actual_indices]  # [num_actual_non_nan]
        positions = torch.searchsorted(
            sorted_sequence=expected_values,
            input=actual_values,
        )  # [num_actual_non_nan]
        within_bounds = positions < expected_values.numel()
        candidates = positions.clamp(max=expected_values.numel() - 1)
        known = within_bounds & (expected_values[candidates] == actual_values)
        mapping[actual_indices[known]] = expected_indices[
            permutation[candidates[known]]
        ]
        return mapping
