# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import pytest
import torch

from sdm import EnsembleTable, TableTensor
from sdm.processing import EnsembleProcessorAdapter, QuantileTransform
from sdm.testing import onlyCUDA, withCUDA


def test_quantile_transform_rejects_nonpositive_n_quantiles() -> None:
    with pytest.raises(ValueError, match="n_quantiles"):
        QuantileTransform(n_quantiles=0)


def test_quantile_transform_rejects_nonpositive_subsample() -> None:
    with pytest.raises(ValueError, match="subsample"):
        QuantileTransform(subsample=0)


@withCUDA
def test_quantile_transform_uniform_fit_transform_and_inverse_round_trip(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = QuantileTransform(n_quantiles=4, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    assert torch.allclose(transformed, expected)
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_quantile_transform_wide_inverse_round_trip(
    device: torch.device,
) -> None:
    inp = torch.linspace(
        -3, 3, steps=64 * 40, dtype=torch.float64, device=device
    ).view(64, 40)

    processor = QuantileTransform(n_quantiles=64, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    inverse = processor.inverse_transform(
        TableTensor.from_tensor(transformed)
    ).numerical

    assert torch.allclose(inverse, inp, atol=1e-8)


@withCUDA
def test_quantile_transform_repeated_values_map_to_midpoint(
    device: torch.device,
) -> None:
    inp = torch.tensor([[0.0], [1.0], [1.0], [2.0]], device=device)

    processor = QuantileTransform(n_quantiles=4, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        transformed.squeeze(1),
        torch.tensor([0.0, 0.5, 0.5, 1.0], device=device),
    )
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_quantile_transform_constant_columns_round_trip(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
        device=device,
    )

    processor = QuantileTransform(n_quantiles=3, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(transformed, torch.zeros_like(inp))
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_quantile_transform_single_quantile_maps_to_zero(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [2.0, 1.0],
            [3.0, 5.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = QuantileTransform(n_quantiles=1, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(transformed, torch.zeros_like(inp))
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp[:1].expand_as(inp),
    )


@withCUDA
def test_quantile_transform_normal_distribution_is_finite_at_bounds(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[-2.0], [-1.0], [0.0], [4.0], [8.0]],
        device=device,
    )

    processor = QuantileTransform(
        n_quantiles=5,
        subsample=None,
        output_distribution="normal",
    ).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert transformed.isfinite().all()
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
        atol=1e-5,
    )


@onlyCUDA
def test_quantile_transform_rejects_mismatched_generator_device() -> None:
    table = TableTensor.from_tensor(torch.rand(8, 2, device="cuda"))

    with pytest.raises(RuntimeError, match="device type for generator"):
        QuantileTransform(subsample=4).fit(
            table,
            generator=torch.Generator(),
        )


def test_quantile_transform_subsample_is_reproducible_with_generator() -> None:
    # Distinct values make the output sensitive to the sampled rows.
    inp = torch.arange(200.0).view(100, 2)

    table = TableTensor.from_tensor(inp)
    first = QuantileTransform(n_quantiles=6, subsample=32).fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )
    second = QuantileTransform(n_quantiles=6, subsample=32).fit_transform(
        table,
        generator=torch.Generator().manual_seed(0),
    )

    assert first.equal(second)


@withCUDA
@pytest.mark.parametrize("subsample", [None, 32])
def test_quantile_transform_adapter_matches_grouped_tables(
    device: torch.device,
    subsample: int | None,
) -> None:
    values = torch.arange(256.0, device=device).view(128, 2)
    contexts = (
        TableTensor.from_tensor(values),
        TableTensor.from_tensor(values.flip(0)),
    )
    query_values = torch.arange(32.0, device=device).view(16, 2) + 0.5
    queries = (
        TableTensor.from_tensor(query_values),
        TableTensor.from_tensor(query_values.flip(0)),
    )
    member_table_ids = (1, 0, 1)
    context = EnsembleTable(
        groups=(cast(TableTensor, torch.stack(contexts)),),
        locations=tuple((0, table_id) for table_id in member_table_ids),
    )
    query = EnsembleTable(
        groups=(cast(TableTensor, torch.stack(queries)),),
        locations=tuple((0, table_id) for table_id in member_table_ids),
    )
    processor = EnsembleProcessorAdapter(
        QuantileTransform(n_quantiles=8, subsample=subsample)
    )

    context_output = processor.fit_transform_ensemble(
        context,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    query_output = processor.transform_ensemble(query)
    restored = processor.inverse_transform_ensemble(context_output)

    expected_contexts = []
    expected_queries = []
    expected_restored = []
    for context_table, query_table in zip(contexts, queries, strict=True):
        reference = QuantileTransform(
            n_quantiles=8,
            subsample=subsample,
        )
        expected_context = reference.fit_transform(
            context_table,
            generator=torch.Generator(device=device).manual_seed(7),
        )
        expected_contexts.append(expected_context)
        expected_queries.append(reference.transform(query_table))
        expected_restored.append(reference.inverse_transform(expected_context))

    for member_id, table_id in enumerate(member_table_ids):
        assert context_output.member(member_id).equal(
            expected_contexts[table_id]
        )
        assert query_output.member(member_id).equal(expected_queries[table_id])
        assert restored.member(member_id).equal(expected_restored[table_id])
