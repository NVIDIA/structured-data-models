from sdm import Stype
from sdm.tensor import TableTensor


def _check_categorical_codes(inp: TableTensor) -> None:
    """Raise if a categorical code exceeds its column's category vocabulary.

    Negative codes encode missing values and are allowed.

    Args:
        inp: The table whose categorical codes are validated.
    """
    columns = inp.columns[Stype.categorical]
    for index, category in enumerate(inp.categorical.categories):
        codes = inp.categorical[..., index]
        if (codes >= category.numel()).any():
            raise ValueError(
                f"Categorical column '{columns[index]}' contains a code "
                "outside its category vocabulary."
            )
