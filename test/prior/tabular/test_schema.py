# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Task
from sdm.prior import LogUniformInt, Uniform, UniformInt
from sdm.prior.tabular import SchemaSampler


@pytest.mark.parametrize(
    ("task_type", "num_classes"),
    [(Task.regression, None), (Task.classification, 5)],
)
def test_schema_sampling(
    task_type: Task,
    num_classes: int | None,
) -> None:
    schema = SchemaSampler(
        task_type=task_type,
        num_columns=10,
        cat_ratio=0.4,
        max_cardinality=2,
        num_classes=num_classes,
    ).sample()

    assert schema.task_type is task_type
    assert schema.target_dim == (1 if num_classes is None else num_classes)
    assert schema.num_columns == 10
    assert schema.column_dims.count(1) == 6
    assert schema.column_dims.count(2) == 4


@pytest.mark.parametrize(("max_cardinality", "bound"), [(20, 9), (5, 5)])
def test_correlated_cardinality_cap(max_cardinality: int, bound: int) -> None:
    schema = SchemaSampler(
        task_type=Task.regression,
        num_columns=10,
        cat_ratio=1.0,
        max_cardinality=max_cardinality,
    ).sample()

    assert all(2 <= cardinality <= bound for cardinality in schema.column_dims)


def test_independent_cardinalities_ignore_correlated_cap() -> None:
    schema = SchemaSampler(
        task_type=Task.regression,
        num_columns=10,
        cat_ratio=1.0,
        max_cardinality=2,
        independent_cat_ratio=1.0,
        max_correlated_cardinality=1,
    ).sample()

    assert schema.column_dims == (2,) * 10


def test_generator_replay_and_cpu_sampling() -> None:
    sampler = SchemaSampler(
        task_type=Task.regression,
        num_columns=UniformInt(2, 100),
        cat_ratio=Uniform(-0.5, 1.2),
        max_cardinality=LogUniformInt(2, 200),
        independent_cat_ratio=Uniform(-0.1, 1.1),
    )
    generator = torch.Generator()
    state = generator.get_state()
    global_state = torch.get_rng_state()
    with torch.device("meta"):
        schemas = [sampler.sample(generator) for _ in range(2)]
    final_state = generator.get_state()
    generator.set_state(state)

    assert schemas == [sampler.sample(generator) for _ in range(2)]
    assert torch.equal(generator.get_state(), final_state)
    assert not torch.equal(state, final_state)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_schema_size_overrides() -> None:
    sampler = SchemaSampler(
        task_type=Task.classification,
        num_columns=UniformInt(2, 3),
        cat_ratio=0.5,
        max_cardinality=2,
        num_classes=UniformInt(2, 3),
    )

    schema = sampler.sample(num_columns=17, num_classes=11)
    assert schema.num_columns == 17
    assert schema.target_dim == 11

    schema = sampler.sample()
    assert 2 <= schema.num_columns <= 3
    assert 2 <= schema.target_dim <= 3
