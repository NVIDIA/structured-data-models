# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import ShuffleColumns


def test_shuffle_columns_renames_to_new_positions() -> None:
    numerical = torch.randn(8, 2)
    code = torch.randint(3, (8, 1))
    x = TableTensor(
        columns={"numerical": ("x_0", "x_2"), "categorical": ("x_1",)},
        numerical=numerical,
        categorical=CategoricalTensor(
            code=code, categories=(torch.arange(3),)
        ),
    )
    generator = torch.Generator()

    output, _ = ShuffleColumns()(x=x, y=x, generator=generator)

    reference = torch.Generator()
    perm = torch.randperm(3, generator=reference).tolist()
    assert generator.get_state().equal(reference.get_state())
    assert output.numerical_columns == (
        f"x_{perm.index(0)}",
        f"x_{perm.index(2)}",
    )
    assert output.categorical_columns == (f"x_{perm.index(1)}",)
    assert output.numerical.equal(numerical)
    assert output.categorical.code.equal(code)
