from typing import cast

import torch

import sdm.processing as sp
from sdm import ColumnarTensor, Stype, TableTensor

_MISSING_MASK_PREFIX = "__kumo_categorical_missing__"


def _track_categorical_missing(table: TableTensor) -> TableTensor:
    missing = table.numerical == -2
    values = table.replace_blocks(
        numerical=table.numerical.masked_fill(missing, float("nan"))
    )
    mask_columns = tuple(
        f"{_MISSING_MASK_PREFIX}{column}"
        for column in table.columns[Stype.numerical]
    )
    masks = TableTensor(
        columns={Stype.id: mask_columns},
        id=ColumnarTensor(missing.unbind(-1)),
    )
    return cast(TableTensor, torch.cat((values, masks), dim=-1))


def _restore_categorical_missing(table: TableTensor) -> TableTensor:
    mask_columns = tuple(
        column
        for column in table.columns[Stype.id]
        if column.startswith(_MISSING_MASK_PREFIX)
    )
    masks = {
        column.removeprefix(_MISSING_MASK_PREFIX): mask
        for column, mask in zip(
            table.columns[Stype.id],
            table.id.unbind(-1),
            strict=True,
        )
        if column.startswith(_MISSING_MASK_PREFIX)
    }
    numerical = table.numerical.clone()
    for index, column in enumerate(table.columns[Stype.numerical]):
        mask = masks.get(column)
        if mask is not None:
            numerical[..., index].masked_fill_(mask, -1.0)
    return table.replace_blocks(numerical=numerical).drop_columns(mask_columns)


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(
                        sort_by="value",
                        min_frequency=2,
                        missing_code=-2,
                    ),
                    sp.ToNumerical(missing_code=-1),
                    _track_categorical_missing,
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    lambda table: table.replace_blocks(
                        numerical=table.numerical.masked_fill(
                            table.numerical.isinf(),
                            float("nan"),
                        )
                    ),
                    sp.DropConstantColumns(),
                    sp.Standardize(epsilon=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(
                        sp.Identity(),
                        sp.PowerTransform(),
                        method="round_robin",
                    ),
                    sp.ClipSigma(threshold=4.0),
                    sp.FlipSign(),
                    sp.ShuffleColumns(method="latin"),
                    sp.SelectColumns(500, method="first"),
                ],
            ),
            _restore_categorical_missing,
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(),
                ],
            ),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=1.0),
            ),
        ],
    )
