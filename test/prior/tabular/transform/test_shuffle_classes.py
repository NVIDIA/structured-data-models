# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import ShuffleClasses


def test_shuffle_classes_permutes_class_codes() -> None:
    x = TableTensor.from_tensor(torch.randn(6, 1))
    code = torch.tensor([[0], [2], [1], [0], [2], [1]])
    labels = TableTensor(
        categorical=CategoricalTensor(code=code, categories=(torch.arange(3),))
    )
    y = TableTensor.from_tensor(torch.randn(6, 1))

    _, shuffled = ShuffleClasses()(x=x, y=labels, generator=torch.Generator())

    perm = torch.randperm(3, generator=torch.Generator())
    assert shuffled.categorical.code.equal(perm[code])
    assert ShuffleClasses()(x=x, y=y)[1] is y
