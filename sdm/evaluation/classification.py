from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor, Stype, TableTensor


def to_class_indices(
    pred: TableTensor,
    target: TableTensor | CategoricalTensor,
) -> tuple[Tensor, Tensor]:
    r"""Convert classification predictions and targets to class-index form.

    Args:
        pred: Prediction table whose numerical columns contain class scores.
            Column names define the class order.
        target: Single-column categorical target.
    """
    if isinstance(target, TableTensor):
        if target.size(-1) != 1:
            raise ValueError(
                f"Expected a single categorical target column "
                f"(got {target.size(-1)} columns)"
            )
        target = target.categorical

    if target.size(-1) != 1:
        raise ValueError(
            f"Expected a single categorical target column "
            f"(got {target.size(-1)} columns)"
        )

    if pred.size()[:-1] != target.size()[:-1]:
        raise ValueError(
            f"Expected prediction and target row dimensions to match "
            f"(got {tuple(pred.size()[:-1])} and "
            f"{tuple(target.size())[:-1]})"
        )

    columns = pred.columns[Stype.numerical]
    pred: Tensor = pred.numerical

    category = target.categories[0]
    code = target.code.squeeze(-1)
    if (code < 0).any():
        raise ValueError("Expected target to not contain missing values")

    if isinstance(category, StringTensor):
        classes = StringTensor.from_list(
            [_parse_column(column, category) for column in columns],
            device=category.device,
        )
    else:
        classes = torch.tensor(
            [_parse_column(column, category) for column in columns],
            dtype=category.dtype,
            device=category.device,
        )
    match = category.unsqueeze(-1) == classes.unsqueeze(0)  # [C_t, C_p]
    if not match.any(dim=-1)[code].all():
        raise ValueError("Target contains classes missing from prediction")

    return pred[..., match.to(torch.int64).argmax(dim=-1)], code


def to_binary_class(
    pred: TableTensor,
    target: TableTensor | CategoricalTensor,
    positive_class: bool | int | float | str,
) -> tuple[Tensor, Tensor]:
    r"""Convert binary predictions and targets to positive-class form.

    Args:
        pred: Prediction table whose numerical columns contain class scores.
            Column names define the class order.
        target: Single-column categorical target.
        positive_class: Class value treated as the positive class.
    """
    if isinstance(target, TableTensor):
        if target.size(-1) != 1:
            raise ValueError(
                f"Expected a single categorical target column "
                f"(got {target.size(-1)} columns)"
            )
        target = target.categorical

    if target.size(-1) != 1:
        raise ValueError(
            f"Expected a single categorical target column "
            f"(got {target.size(-1)} columns)"
        )

    if pred.size()[:-1] != target.size()[:-1]:
        raise ValueError(
            f"Expected prediction and target row dimensions to match "
            f"(got {tuple(pred.size()[:-1])} and "
            f"{tuple(target.size())[:-1]})"
        )

    score = pred[str(positive_class)].numerical.squeeze(-1)
    match = cast(Tensor, target.categories[0] == positive_class)
    index = match.nonzero().view(-1)
    if index.numel() != 1:
        raise ValueError(
            f"Expected positive class {positive_class!r} to appear exactly "
            f"once in target categories (found {index.numel()} matches)"
        )

    return score, target.code.squeeze(-1) == index


def _parse_column(
    column: str,
    category: Tensor,
) -> bool | int | float | str:
    if isinstance(category, StringTensor):
        return column
    if category.dtype == torch.bool:
        if column == "True":
            return True
        if column == "False":
            return False
        raise ValueError(f"Cannot parse prediction column {column!r} as bool")
    if category.is_floating_point():
        return float(column)
    return int(column)
