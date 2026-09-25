# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import torch

import sdm


@pytest.fixture
def benchmark_model(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    pytest.importorskip("autogluon.core.models.abstract.shared_weights")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    return importlib.import_module("benchmark.tabular.model")


@pytest.mark.parametrize(
    ("columns", "values", "expected"),
    [
        (("0", "1", "2"), [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]),
        (("2", "0", "1"), [0.7, 0.1, 0.2], [0.1, 0.2, 0.7]),
    ],
)
def test_adapter_classification_output_alignment(
    benchmark_model: ModuleType,
    columns: tuple[str, ...],
    values: list[float],
    expected: list[float],
) -> None:
    output = sdm.TableTensor(
        columns={sdm.Stype.numerical: columns},
        numerical=torch.tensor([values], dtype=torch.float16),
    )
    adapter = object.__new__(benchmark_model.SDMKumoTabularModel)
    adapter.model = type("Model", (), {"predict": lambda self, x: output})()
    adapter._device = torch.device("cpu")
    adapter._expand_query = False
    adapter.stypes = {"x": sdm.Stype.numerical}
    adapter.problem_type = "multiclass"
    adapter.num_classes = 3
    adapter.preprocess = lambda frame, **kwargs: frame
    adapter._convert_proba_to_unified_form = lambda probabilities: (
        probabilities
    )

    probabilities = adapter._predict_proba(pd.DataFrame({"x": [1.0]}))

    torch.testing.assert_close(
        torch.from_numpy(probabilities),
        torch.tensor([expected], dtype=torch.float16).float(),
    )


def test_adapter_regression_uses_float32_reduction(
    benchmark_model: ModuleType,
) -> None:
    values = torch.tensor(
        [[1.0, 2.0, 4.0]],
        dtype=torch.float16,
    )
    output = sdm.TableTensor.from_tensor(values)
    adapter = object.__new__(benchmark_model.SDMKumoTabularModel)
    adapter.model = type("Model", (), {"predict": lambda self, x: output})()
    adapter._device = torch.device("cpu")
    adapter._expand_query = False
    adapter.stypes = {"x": sdm.Stype.numerical}
    adapter.problem_type = "regression"
    adapter.preprocess = lambda frame, **kwargs: frame

    prediction = adapter._predict_proba(pd.DataFrame({"x": [1.0]}))

    assert prediction.dtype.name == "float32"
    torch.testing.assert_close(
        torch.from_numpy(prediction),
        values.mean(dim=-1, dtype=torch.float32),
    )
