# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm


def test_to_quantiles() -> None:
    pred = sdm.TableTensor.from_tensor(
        torch.randn(2, 5, 4),
        columns=["b__q10", "b__q90", "a__q10", "a__q90"],
    )
    values, levels = sdm.evaluation.to_quantiles(pred)
    torch.testing.assert_close(values, pred.numerical.view(2, 5, 2, 2))
    torch.testing.assert_close(levels, torch.tensor([0.1, 0.9]))


def test_to_quantiles_invalid() -> None:
    pred = sdm.TableTensor.from_tensor(
        torch.randn(5, 4),
        columns=["a__q10", "b__q10", "a__q90", "b__q90"],
    )
    with pytest.raises(ValueError, match="grouped by target"):
        sdm.evaluation.to_quantiles(pred)
