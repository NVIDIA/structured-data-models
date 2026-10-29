# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
import torch
from torch import Tensor

import sdm.processing as sp
from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel, KumoRelational, KumoTabular, TabFM, TabICLv2
from sdm.models.kumo.tabular.row_embedding import (
    RowEmbedding as KumoTabularRowEmbedding,
)
from sdm.models.tabfm.row_embedding import RowEmbedding as TabFMRowEmbedding
from sdm.models.tabiclv2.row_embedding import (
    RowEmbedding as TabICLv2RowEmbedding,
)
from sdm.testing import onlyCUDA


def _randomize(module: torch.nn.Module) -> torch.nn.Module:
    # Zero-initialized residual branches would hide the encoder layers.
    for parameter in module.parameters():
        if not parameter.any():
            torch.nn.init.normal_(parameter, std=0.02)
    return module


def _kumo_tabular(device: torch.device) -> Callable[..., Tensor]:
    module = _randomize(
        KumoTabularRowEmbedding(
            num_classes=3,
            channels=16,
            num_layers=2,
            num_heads=2,
            group_size=3,
            num_frequencies=8,
            num_inducing_points=4,
            num_readout_tokens=2,
            device=device,
        )
    )
    return lambda x, y, **kwargs: module(
        x,
        y,
        torch.zeros(x.size(-1), dtype=torch.bool, device=device),
        **kwargs,
    )


def _tabfm(device: torch.device) -> Callable[..., Tensor]:
    module = _randomize(
        TabFMRowEmbedding(
            num_classes=3,
            channels=16,
            num_layers=2,
            num_repeats=2,
            num_col_heads=2,
            num_row_heads=2,
            group_size=3,
            num_frequencies=8,
            num_inducing_points=4,
            num_readout_tokens=2,
            device=device,
        )
    )
    return lambda x, y, **kwargs: module(
        x,
        y,
        torch.zeros(x.size(-1), dtype=torch.bool, device=device),
        **kwargs,
    )


def _tabiclv2(device: torch.device) -> Callable[..., Tensor]:
    module = _randomize(
        TabICLv2RowEmbedding(
            num_classes=3,
            channels=16,
            num_layers=2,
            num_heads=2,
            group_size=3,
            num_inducing_points=4,
            num_readout_tokens=2,
            norm_bias=True,
            device=device,
        )
    )
    return lambda x, y, **kwargs: module(x, y, num_classes=3, **kwargs)


@onlyCUDA
@pytest.mark.parametrize("build", [_kumo_tabular, _tabfm, _tabiclv2])
@pytest.mark.parametrize("chunk_memory_fraction", ["0.05", "1e-9"])
def test_row_embedding_offload(
    build: Callable[[torch.device], Callable[..., Tensor]],
    chunk_memory_fraction: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", chunk_memory_fraction)
    device = torch.device("cuda")
    embedding = build(device)
    x = torch.randn(1, 40, 5, device=device)
    y = torch.randint(0, 3, (1, 30), device=device)

    outs: dict[Any, Tensor] = {}
    with torch.no_grad():
        for offload in (None, "cpu"):
            cache = Cache()
            outs[offload, "fit"] = embedding(
                x[:, :30], y, cache=cache, offload=offload
            )
            outs[offload, "predict"] = embedding(
                x[:, 30:], y[:, :0], cache=cache.freeze()
            )
            outs[offload] = embedding(x, y, offload=offload)

    for key in ("fit", "predict"):
        torch.testing.assert_close(outs["cpu", key], outs[None, key])
    torch.testing.assert_close(outs["cpu"], outs[None])


@onlyCUDA
def test_row_embedding_offload_subsamples_keys() -> None:
    device = torch.device("cuda")
    embedding = _tabiclv2(device)
    x = torch.randn(1, 40, 5, device=device)
    y = torch.randint(0, 3, (1, 30), device=device)

    with torch.no_grad():
        outs = [
            embedding(
                x,
                y,
                max_keys=10,
                generator=torch.Generator(device).manual_seed(0),
                offload=offload,
            )
            for offload in (None, "cpu")
        ]

    torch.testing.assert_close(outs[1], outs[0])


def test_offload_requires_cuda_inputs() -> None:
    model = KumoTabular(task="classification", size="small", pretrained=False)
    x = TableTensor.from_tensor(torch.randn(8, 5))
    y = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.randint(0, 3, (6, 1)),
            categories=(torch.arange(3),),
        ),
    )

    with pytest.raises(ValueError, match="CUDA"):
        model.fit(x[:6], y, offload="cpu")
    with pytest.raises(ValueError, match="CUDA"):
        model(x[:6], y, x[6:], offload="cpu")


def test_offload_requires_model_support() -> None:
    model = KumoRelational(pretrained=False)
    x = TableTensor.from_tensor(torch.randn(6, 5))
    y = TableTensor.from_tensor(torch.randn(6, 1))

    with pytest.raises(ValueError, match="does not support 'offload'"):
        model.fit(x, y, offload="cpu")


@onlyCUDA
@pytest.mark.parametrize(
    "model_cls",
    [partial(KumoTabular, size="small"), TabFM, TabICLv2],
)
def test_fit_predict_offload(model_cls: Callable[..., ICLModel]) -> None:
    device = torch.device("cuda")
    model = model_cls(task="classification", pretrained=False, device=device)
    _randomize(model)
    x = TableTensor.from_tensor(torch.randn(48, 5, device=device))
    y = TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.randint(0, 3, (32, 1), device=device),
            categories=(torch.arange(3, device=device),),
        ),
    )
    recipe = sp.Recipe(
        features=[sp.ToNumerical()],
        output=[sp.AverageEstimators()],
    )

    outs = []
    for offload in (None, "cpu"):
        model.fit(x[:32], y, recipe=recipe, num_estimators=2, offload=offload)
        outs.append(model.predict(x[32:]))

    torch.testing.assert_close(outs[1].numerical, outs[0].numerical)

    with pytest.raises(ValueError, match="estimator_batch_size"):
        model.fit(
            x[:32],
            y,
            recipe=recipe,
            num_estimators=2,
            estimator_batch_size=None,
            offload="cpu",
        )
