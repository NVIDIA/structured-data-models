from sdm import Stype
from sdm.tensor import TableTensor


def _check_categorical_codes(input: TableTensor) -> None:
    columns = input.columns[Stype.categorical]
    for index, category in enumerate(input.categorical.categories):
        codes = input.categorical.as_tensor()[..., index]
        if (codes >= category.numel()).any():
            raise ValueError(
                f"Categorical column '{columns[index]}' contains a code "
                "outside its category vocabulary."
            )
