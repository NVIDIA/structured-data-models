from sdm import Stype, TableTensor


def _check_categorical_codes(table: TableTensor) -> None:
    """Raise if a categorical code exceeds its column's category vocabulary.

    Negative codes encode missing values and are allowed.

    Args:
        table: The table whose categorical codes are validated.
    """
    codes = table.categorical.code
    if codes.numel() == 0:
        return
    bounds = codes.new_tensor(
        [category.numel() for category in table.categorical.categories]
    )
    invalid = codes.amax(dim=tuple(range(codes.dim() - 1))) >= bounds
    if invalid.any():
        index = int(invalid.nonzero()[0, 0].item())
        columns = table.columns[Stype.categorical]
        raise ValueError(
            f"Categorical column {columns[index]!r} contains a code "
            "outside its category vocabulary."
        )
