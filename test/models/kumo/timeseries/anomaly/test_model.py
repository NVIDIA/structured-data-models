# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import sdm.processing as sp
from sdm import CategoricalTensor, Recipe, Stype, TableTensor
from sdm.explain.gradient import CaptureInputs, EnableInputGradients
from sdm.models import Callback, KumoAnomaly
from sdm.testing import withCUDA


@withCUDA
def test_forward(device: torch.device) -> None:
    model = KumoAnomaly(pretrained=False, device=device)
    x = torch.randn(2, 130, 3, device=device)

    def generator() -> torch.Generator:
        return torch.Generator(device).manual_seed(0)

    out = model(x, num_samples=1, num_steps=1, generator=generator())
    assert out.size() == (2, 130, 1)
    assert out.columns["numerical"] == ("mae",)
    assert out.device == device
    assert out.numerical.isfinite().all()
    assert (out.numerical >= 0).all()

    again = model(x, num_samples=1, num_steps=1, generator=generator())
    assert torch.equal(again.numerical, out.numerical)


def test_wide_float64_forward() -> None:
    model = KumoAnomaly(pretrained=False)
    x = torch.randn(60, 50, dtype=torch.float64)

    out = model(x, num_samples=1, num_steps=1)
    assert out.size() == (60, 1)
    assert out.numerical.dtype == torch.float64


@pytest.mark.parametrize("batch_shape", [(), (2,)])
@pytest.mark.parametrize("reduce_members", [False, True])
def test_output_recipe(
    batch_shape: tuple[int, ...], reduce_members: bool
) -> None:
    model = KumoAnomaly(pretrained=False)
    x = torch.randn(*batch_shape, 12, 2)
    generator = torch.Generator()
    state = generator.get_state()
    expected = model(x, num_samples=1, num_steps=1, generator=generator)
    recipe = model.default_recipe()
    if reduce_members:
        recipe.output = sp.AverageEstimators()
    recipe.append_output(
        lambda table: table.replace_blocks(numerical=table.numerical + 2)
    )
    generator.set_state(state)
    out = model(
        x=x,
        recipe=recipe,
        num_samples=1,
        num_steps=1,
        generator=generator,
    )
    assert out.numerical_columns == ("mae",)
    torch.testing.assert_close(out.numerical, expected.numerical + 2)


class DoubleScores(Callback):
    def on_model_forward_end(
        self,
        model: torch.nn.Module,
        out: TableTensor,
    ) -> TableTensor:
        return out.replace_blocks(numerical=2 * out.numerical.double())


def test_callbacks() -> None:
    model = KumoAnomaly(pretrained=False)
    x = torch.randn(30, 2)
    capture = CaptureInputs()
    callbacks = [EnableInputGradients(), capture, DoubleScores()]

    def generator() -> torch.Generator:
        return torch.Generator().manual_seed(0)

    out = model(x, num_samples=1, num_steps=1, generator=generator())
    doubled = model(
        x,
        num_samples=1,
        num_steps=1,
        callbacks=callbacks,
        generator=generator(),
    )
    assert doubled.numerical.dtype == torch.float32
    torch.testing.assert_close(doubled.numerical, 2 * out.numerical)

    doubled.numerical.sum().backward()
    ((x, _),) = capture.inputs
    assert x.numerical.grad is not None


def test_invalid_inputs() -> None:
    model = KumoAnomaly(pretrained=False)
    x = TableTensor(
        columns={Stype.numerical: ["a"], Stype.categorical: ["b"]},
        numerical=torch.randn(10, 1),
        categorical=CategoricalTensor.from_tensor(
            torch.randint(0, 5, (10, 1))
        ),
    )

    with pytest.warns(RuntimeWarning, match="unsupported feature stypes"):
        model(x, num_samples=1, num_steps=1)
    with pytest.raises(ValueError, match="at most 40 features"):
        model(torch.randn(10, 41), recipe=Recipe())
