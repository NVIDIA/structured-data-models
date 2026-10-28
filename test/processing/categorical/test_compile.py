# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import CategoricalTensor, StringTensor, TableTensor, VarLenTensor
from sdm.processing import AddCategoryCounts, ImputeMode
from sdm.testing import withCUDA


def _table(category: torch.Tensor, codes: list[int]) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("city",)},
        categorical=CategoricalTensor(
            code=torch.tensor(codes, device=category.device).unsqueeze(-1),
            categories=(category,),
        ),
    )


@withCUDA
@pytest.mark.parametrize(
    ("processor_type", "category"),
    [
        (ImputeMode, torch.tensor([1, 0, 2, 0])),
        (ImputeMode, VarLenTensor.from_list([[1], [], [2, 3], []])),
        (AddCategoryCounts, StringTensor.from_list(["a", "", "b", ""])),
    ],
    ids=["tensor", "var_len", "string"],
)
def test_category_checks_compile(
    compilation: None,
    device: torch.device,
    processor_type: type[ImputeMode | AddCategoryCounts],
    category: torch.Tensor,
) -> None:
    category = category.to(device)[::2]
    table = _table(category, [0, 1, -1, -7])
    processor = processor_type().fit(_table(category.clone(), [0, 1, -1, -7]))
    compiled = torch.compile(processor.transform, fullgraph=True)

    assert compiled(table).equal(processor.transform(table))
    with pytest.raises(ValueError, match="outside its category vocabulary"):
        compiled(_table(category, [0, 2, -1, -7]))
    reordered = category[torch.tensor([1, 0], device=device)]
    with pytest.raises(ValueError, match="match the fitted values and order"):
        compiled(_table(reordered, [0, 1, -1, -7]))
