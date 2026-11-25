# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor
from sdm.evaluation import to_class_indices


def _to_table(values: Tensor, like: TableTensor) -> TableTensor:
    # Invert `ToNumerical`: return features `[R, D]`, whose categorical
    # columns hold codes, as a table with the columns of `like`.
    numerical, code = values.tensor_split([like.numerical.size(-1)], dim=-1)
    # Casting NaN to an integer gives a valid code, so replace it first:
    code = code.nan_to_num(nan=-1.0).to(like.categorical.code.dtype)
    return TableTensor(
        columns={
            "numerical": like.numerical_columns,
            "categorical": like.categorical_columns,
        },
        numerical=numerical.to(like.numerical.dtype),
        categorical=CategoricalTensor(code, like.categorical.categories),
    )


def _score(pred: TableTensor, target: TableTensor) -> Tensor:
    # Return the default score of predictions `[R, O]` of the targets
    # `[R, 1]`, which is the negative loss.
    if target.categorical.size(-1) > 0:  # Log loss, with classes by name.
        prob, index = to_class_indices(pred, target, missing_score=0.0)
        prob = prob.gather(-1, index.unsqueeze(-1)).squeeze(-1)  # [R]
        return prob.clamp_min(torch.finfo(prob.dtype).tiny).log().mean()
    # Squared error of the mean over all predicted columns:
    mean = pred.numerical.mean(dim=-1)  # [R]
    return -(mean - target.numerical[:, 0]).square().mean()
