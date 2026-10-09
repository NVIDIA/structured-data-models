# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch

import sdm
from benchmark.tabular import finetune
from sdm.testing import withCUDA


@pytest.mark.parametrize("epochs", [None, 1, 75])
@pytest.mark.parametrize("is_kumo_small", [False, True])
@pytest.mark.parametrize("is_binary", [False, True])
def test_epochs(
    epochs: int | None,
    is_kumo_small: bool,
    is_binary: bool,
) -> None:
    expected = epochs
    if expected is None:
        expected = 50 if is_kumo_small and is_binary else 75
    assert (
        finetune.kumo_small_binary_epochs(
            epochs,
            is_kumo_small=is_kumo_small,
            is_binary=is_binary,
        )
        == expected
    )


class _Regressor(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), device=device))
        self.extra_state = {"updates": 0}

    def forward(self, x_query: sdm.TableTensor, **_: Any) -> sdm.TableTensor:
        if self.training:
            self.extra_state["updates"] += 1
        return sdm.TableTensor.from_tensor(
            self.weight.expand(x_query.size(0), 1)
        )

    def get_extra_state(self) -> dict[str, int]:
        return self.extra_state

    def set_extra_state(self, state: dict[str, int]) -> None:
        self.extra_state = state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        assert all(
            value.device.type == "cpu"
            for value in state_dict.values()
            if isinstance(value, torch.Tensor)
        )
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )


@withCUDA
@pytest.mark.parametrize("improve", [False, True])
def test_best_checkpoint(
    device: torch.device,
    improve: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Regressor(device)
    weights: list[torch.Tensor] = []
    metrics = iter([3.0, 2.0 if improve else 4.0, 5.0])

    def evaluate(*_: Any, **__: Any) -> float:
        model.eval()
        weights.append(model.weight.detach().clone())
        return next(metrics)

    monkeypatch.setattr(finetune, "evaluate", evaluate)
    x = sdm.TableTensor.from_tensor(torch.ones(10, 1, device=device))
    y = sdm.TableTensor.from_tensor(torch.zeros(10, 1, device=device))
    metric = finetune.full_finetune(
        cast(sdm.models.ICLModel, model),
        x,
        y,
        task="regression",
        max_epochs=2,
        iters_per_epoch=1,
        train_size=8,
        context_frac=0.5,
        val_frac=0.2,
        lr=0.1,
        num_estimators=1,
        generator=None,
    )
    assert metric == (2.0 if improve else 3.0)
    torch.testing.assert_close(model.weight, weights[int(improve)])
    assert not torch.equal(weights[0], weights[-1])
    assert model.extra_state == {"updates": int(improve)}
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
