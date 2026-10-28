# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
    VarLenTensor,
)
from sdm.processing import AddCategoryCounts, ImputeMode
from sdm.testing import withCUDA


def _table(category: torch.Tensor, codes: list[int]) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("city",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                data=codes, dtype=torch.int32, device=category.device
            ).unsqueeze(-1),
            categories=(category,),
        ),
    )


@withCUDA
@pytest.mark.parametrize("processor_type", [ImputeMode, AddCategoryCounts])
@pytest.mark.parametrize("kind", ["tensor", "var_len", "string"])
def test_category_checks_compile(
    compilation: None,
    device: torch.device,
    processor_type: type[ImputeMode | AddCategoryCounts],
    kind: str,
) -> None:
    if kind == "tensor":
        category = torch.tensor([1, 2], device=device)
    elif kind == "var_len":
        category = VarLenTensor.from_list([[1], [2, 3]], device=device)
    else:
        # A vocabulary view retains its physical storage and logical stride.
        category = StringTensor.from_list(
            values=["a", "unused", "b", "unused"], device=device
        )[::2]
    table = _table(category, [0, 1, -1, -7])
    processor = processor_type().fit(_table(category.clone(), [0, 1, -1, -7]))
    compiled = torch.compile(processor.transform, fullgraph=True)

    assert compiled(table).equal(processor.transform(table))
    with pytest.raises(ValueError, match="outside its category vocabulary"):
        compiled(_table(category, [0, 2, -1, -7]))
    reordered = category[torch.tensor([1, 0], device=device)]
    with pytest.raises(ValueError, match="match the fitted values and order"):
        compiled(_table(reordered, [0, 1, -1, -7]))
