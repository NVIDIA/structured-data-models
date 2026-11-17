# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm.prior import (
    HP,
    Correlated,
    CorrelatedContext,
    Mixture,
    Uniform,
    UniformInt,
)


def test_correlated_sampling() -> None:
    distribution = Correlated(Uniform(0.0, 1.0))
    assert HP.from_value(distribution) is distribution
    generator = torch.Generator()
    state = generator.get_state()
    global_state = torch.get_rng_state()

    with torch.device("meta"):
        values = distribution.sample_batch(8, generator)
    assert values.shape == (8,)
    assert values.device == torch.device("cpu")
    generator.set_state(state)
    assert values.equal(distribution.sample_batch(8, generator))
    assert torch.equal(torch.get_rng_state(), global_state)
    assert isinstance(distribution.sample(generator), float)
    assert isinstance(Correlated(UniformInt(1, 10)).sample(generator), int)


def test_correlated_shared_key() -> None:
    distribution = Correlated(Uniform(0.0, 1.0), key="width")
    shifted = Correlated(Uniform(10.0, 20.0), key="width")
    generator = torch.Generator()

    with CorrelatedContext(generator=generator):
        distribution.sample_batch((2, 8))
        state = generator.get_state()
        values = distribution.sample_batch((2, 8))
        final_state = generator.get_state()
        generator.set_state(state)
        mapped = shifted.sample_batch((2, 8))

    assert not torch.equal(state, final_state)
    torch.testing.assert_close(mapped, 10.0 + 10.0 * values)
    assert torch.equal(generator.get_state(), final_state)


def test_context_reuse_starts_fresh() -> None:
    generator = torch.Generator()
    context = CorrelatedContext(generator=generator)
    distribution = Correlated(Uniform(0.0, 1.0))
    state = generator.get_state()

    with context:
        values = distribution.sample_batch(8)
    final_state = generator.get_state()
    generator.set_state(state)
    with context:
        assert values.equal(distribution.sample_batch(8))
    assert torch.equal(generator.get_state(), final_state)


def test_correlated_mixture_shared_key() -> None:
    items = [("sum", 4.0), ("prod", 1.0)]
    choice = Mixture(items, correlated=True, key="fn")
    alias = Mixture(items, correlated=True, key="fn")
    generator = torch.Generator()

    with CorrelatedContext(generator=generator):
        choice.pick()
        state = generator.get_state()
        values = [choice.pick() for _ in range(8)]
        final_state = generator.get_state()
        assert not torch.equal(state, final_state)
        generator.set_state(state)
        assert values == [alias.pick() for _ in range(8)]
        assert torch.equal(generator.get_state(), final_state)
