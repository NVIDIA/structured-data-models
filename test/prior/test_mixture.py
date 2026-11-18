# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm.prior import CorrelatedContext, Mixture


@pytest.mark.parametrize(
    "weights", [(), (0.0, 0.0), (-1.0, -1.0), (-1.0, 1.0)]
)
def test_invalid_weights(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        Mixture([(i, weight) for i, weight in enumerate(weights)])


def test_generator_conflicts() -> None:
    generator = torch.Generator()
    other = torch.Generator()
    state = generator.get_state()
    other_state = other.get_state()
    mixture = Mixture(["sum", "prod"], True)

    with CorrelatedContext(generator=generator):
        with pytest.raises(ValueError):  # noqa: PT011
            mixture.pick(other)
        assert torch.equal(generator.get_state(), state)
        assert torch.equal(other.get_state(), other_state)

        assert mixture.pick(generator) in {"sum", "prod"}
        assert mixture.pick() in {"sum", "prod"}
