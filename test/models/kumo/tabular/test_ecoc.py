# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models import ECOC
from sdm.models.kumo.tabular import KumoTabular
from sdm.models.kumo.tabular.model import MODEL_KWARGS


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> KumoTabular:
    monkeypatch.setitem(
        MODEL_KWARGS,
        "small",
        {
            "cell_channels": 8,
            "num_embedding_layers": 1,
            "num_embedding_heads": 2,
            "num_inducing_points": 2,
            "group_size": 2,
            "num_frequencies": 2,
            "num_readout_tokens": 2,
            "icl_channels": 16,
            "num_icl_layers": 2,
            "num_icl_heads": 2,
        },
    )
    model = KumoTabular(task="classification", size="small", pretrained=False)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    return model


@pytest.mark.parametrize("num_classes", [10, 12, 23])
def test_many_class_fit_predict(model: KumoTabular, num_classes: int) -> None:
    x = TableTensor.from_tensor(torch.randn(num_classes, 3))
    query = TableTensor.from_tensor(torch.randn(5, 3))
    y = TableTensor(
        columns={Stype.categorical: ["target"]},
        categorical=CategoricalTensor(
            code=torch.arange(num_classes).view(-1, 1),
            categories=[torch.arange(num_classes) * 7 + 5],
        ),
    )

    expected = model(
        x,
        y,
        query,
        num_estimators=2,
        generator=torch.Generator().manual_seed(0),
    )
    model.fit(
        x, y, num_estimators=2, generator=torch.Generator().manual_seed(0)
    )
    actual = model.predict(query)

    assert actual.shape == (5, num_classes)
    assert set(actual.columns[Stype.numerical]) == {
        str(i * 7 + 5) for i in range(num_classes)
    }
    assert actual.columns == expected.columns
    torch.testing.assert_close(actual.numerical, expected.numerical)
    torch.testing.assert_close(actual.numerical.sum(dim=-1), torch.ones(5))
    assert torch.is_inference(actual)

    repeated = model.predict(query[:2])
    torch.testing.assert_close(repeated.numerical, actual.numerical[:2])


def test_pretrained_checkpoint(
    model: KumoTabular,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Published checkpoints contain the underlying model without the wrapper.
    path = tmp_path / "classifier.pt"
    classifier = cast(ECOC, model.models["classification"])
    torch.save({"model": classifier.model.state_dict()}, path)
    monkeypatch.setattr(
        "sdm.models.kumo.tabular.model.download_checkpoint",
        lambda **kwargs: path,
    )
    loaded = KumoTabular(task="classification", size="small", pretrained=True)
    x = torch.randn(15, 3)
    y = torch.arange(12).view(-1, 1)

    expected = model(
        x[:12], y, x[12:], generator=torch.Generator().manual_seed(0)
    )
    actual = loaded(
        x[:12], y, x[12:], generator=torch.Generator().manual_seed(0)
    )

    torch.testing.assert_close(actual.numerical, expected.numerical)
