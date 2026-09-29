# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.processing import RankGaussian
from sdm.testing import onlyCUDA, withCUDA


def edge_query(context: Tensor) -> Tensor:
    # [..., 14, C] queries at, between and beyond the fitted values.
    values = context.new_tensor(
        [
            -torch.inf,
            -100.0,
            -7.5,
            -4.0,
            -0.0,
            0.0,
            1.0,
            2.5,
            3.0,
            6.0,
            7.5,
            100.0,
            torch.inf,
            torch.nan,
        ]
    )
    return values[:, None].expand(*context.shape[:-2], -1, context.size(-1))


@withCUDA
def test_mid_ranks_and_interpolated_query(device: torch.device) -> None:
    context = torch.tensor(
        [[0.0], [0.0], [1.0], [1.0], [2.0], [2.0]],
        dtype=torch.float64,
        device=device,
    )
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    probabilities = context.new_tensor(
        [[1 / 6], [1 / 6], [0.5], [0.5], [5 / 6], [5 / 6]]
    )
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(context)).numerical,
        torch.special.ndtri(probabilities),
    )
    query = context.new_tensor([[-100.0], [0.5], [1.5], [100.0], [torch.nan]])
    expected = context.new_tensor(
        [[1 / 6], [1 / 3], [2 / 3], [5 / 6], [torch.nan]]
    )
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(query)).numerical,
        torch.special.ndtri(expected),
        equal_nan=True,
    )


@withCUDA
def test_constants_and_missing_columns(device: torch.device) -> None:
    context = torch.tensor([[7.0, torch.nan]], device=device).expand(4, -1)
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    query = context.new_tensor(
        [[7.0, 8.0], [torch.nan, torch.nan], [100.0, 3.0]]
    )
    expected = context.new_tensor(
        [[0.0, torch.nan], [torch.nan, torch.nan], [0.0, torch.nan]]
    )
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(query)).numerical,
        expected,
        equal_nan=True,
    )


@withCUDA
def test_batched_missing_values_match_independent_columns(
    device: torch.device,
) -> None:
    context = torch.randn(2, 3, 9, 4, device=device)
    context[..., 1, 0] = torch.nan
    context[..., :3, 1] = torch.nan
    context[..., :, 2] = 2.0
    context[..., :, 3] = torch.nan
    query = torch.randn(2, 3, 5, 4, device=device)
    query[..., 0, 0] = torch.nan
    processor = RankGaussian(max_knots=4).fit(TableTensor.from_tensor(context))
    output = processor.transform(TableTensor.from_tensor(query)).numerical
    for train, test, actual in zip(
        context.flatten(0, 1),
        query.flatten(0, 1),
        output.flatten(0, 1),
        strict=True,
    ):
        reference = RankGaussian(max_knots=4).fit(
            TableTensor.from_tensor(train)
        )
        expected = reference.transform(TableTensor.from_tensor(test)).numerical
        torch.testing.assert_close(actual, expected, equal_nan=True)


@withCUDA
def test_few_distinct_values_match_all_knots(device: torch.device) -> None:
    context = torch.tensor(
        [
            [0, 0, 0, torch.nan],
            [0, -0.0, 0, torch.nan],
            [0, -1, 0, torch.nan],
            [1, -1, 0, torch.nan],
            [1, -1, 2, torch.nan],
            [1, -2, 4, torch.nan],
            [2, -2, 6, torch.nan],
            [2, -2, 6, torch.nan],
            [2, -3, 6, torch.nan],
            [3, -3, 6, torch.nan],
        ],
        dtype=torch.float64,
        device=device,
    )
    table = TableTensor.from_tensor(context)
    query = TableTensor.from_tensor(edge_query(context))
    processor = RankGaussian(max_knots=4)
    reference = RankGaussian()
    torch.testing.assert_close(
        actual=processor.fit_transform(table).numerical,
        expected=reference.fit_transform(table).numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        actual=processor.transform(query).numerical,
        expected=reference.transform(query).numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@withCUDA
@pytest.mark.parametrize("max_knots", [2, 16])
def test_many_distinct_values_stay_within_one_knot_spacing(
    max_knots: int, device: torch.device
) -> None:
    # For columns without ties, these knot counts bound the error by the knot
    # spacing whatever the values.
    context = torch.randn(2, 300, 2, dtype=torch.float64, device=device)
    context[..., 1] = context[..., 1].mul(2).exp()
    table = TableTensor.from_tensor(context)
    processor = RankGaussian(max_knots=max_knots).fit(table)
    reference = RankGaussian().fit(table)
    output = processor.transform(table).numerical
    expected = reference.transform(table).numerical

    lower = expected.amin(dim=-2, keepdim=True)
    upper = expected.amax(dim=-2, keepdim=True)
    spacing = (upper - lower) / (max_knots - 1)
    assert (output - expected).abs().le(spacing).all()

    # The extreme fitted values keep their scores, and queries beyond clamp:
    smallest = context.amin(dim=-2, keepdim=True)
    largest = context.amax(dim=-2, keepdim=True)
    beyond = torch.cat(
        [
            smallest,
            largest,
            smallest - 1,
            largest + 1,
            torch.full_like(smallest, -torch.inf),
            torch.full_like(largest, torch.inf),
        ],
        dim=-2,
    )
    torch.testing.assert_close(
        actual=processor.transform(TableTensor.from_tensor(beyond)).numerical,
        expected=torch.cat([lower, upper] * 3, dim=-2),
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    values = context.sort(dim=-2).values
    midpoints = values[..., :-1, :].lerp(values[..., 1:, :], 0.5)
    queries = torch.cat([values, midpoints, beyond], dim=-2)
    scores = processor.transform(
        TableTensor.from_tensor(queries.sort(dim=-2).values)
    ).numerical
    assert scores.diff(dim=-2).ge(0).all()


@withCUDA
@pytest.mark.parametrize("max_knots", [16, None])
def test_nonfinite_context_does_not_change_fitted_ranks(
    max_knots: int | None,
    device: torch.device,
) -> None:
    finite = torch.randn(200, 2, dtype=torch.float64, device=device)
    nonfinite = finite.new_tensor([torch.nan, torch.inf, -torch.inf])
    context = torch.cat([finite, nonfinite[:, None].repeat(20, 2)])
    context = context[torch.randperm(context.size(0), device=device)]
    query = torch.cat([context, edge_query(context)])
    processor = RankGaussian(max_knots=max_knots).fit(
        TableTensor.from_tensor(context)
    )
    reference = RankGaussian(max_knots=max_knots).fit(
        TableTensor.from_tensor(finite)
    )
    output = processor.transform(TableTensor.from_tensor(query)).numerical
    torch.testing.assert_close(
        actual=output,
        expected=reference.transform(TableTensor.from_tensor(query)).numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.equal(output.isnan(), query.isnan())
    assert output[~query.isnan()].isfinite().all()


@withCUDA
def test_values_next_to_ties_stay_within_one_knot_spacing(
    device: torch.device,
) -> None:
    # 500 zeros, then 500 distinct values.
    context = torch.cat([torch.zeros(500), torch.arange(1.0, 501.0)]).to(
        dtype=torch.float64, device=device
    )[:, None]
    table = TableTensor.from_tensor(context)
    output = RankGaussian(max_knots=64).fit_transform(table).numerical
    reference = RankGaussian()
    expected = reference.fit_transform(table).numerical
    spacing = (expected.max() - expected.min()) / 63
    assert (output - expected).abs().le(spacing).all()


@onlyCUDA
def test_chunks_match_single_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    context = torch.randn(2, 300, 5, device="cuda")
    context[..., 1] = context[..., 1].mul(2).round()
    context[:, ::7, 0] = torch.nan
    context[:, 3, 1] = torch.inf
    context[..., 3] = 1.0
    context[..., 4] = torch.nan
    table = TableTensor.from_tensor(context)
    query = torch.randn(2, 40, 5, device="cuda") * 3
    query[:, ::6] = torch.nan
    query = TableTensor.from_tensor(
        torch.cat([query, edge_query(query)], dim=-2)
    )

    # Chunks run without autograd:
    with torch.inference_mode():
        processor = RankGaussian(max_knots=16)
        expected = processor.fit_transform(table)
        expected_query = processor.transform(query)
        # Chunks of a single column in fit and a single row in transform:
        monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", "1e-12")
        processor = RankGaussian(max_knots=16)
        output = processor.fit_transform(table)
        output_query = processor.transform(query)

    torch.testing.assert_close(
        actual=output.numerical,
        expected=expected.numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        actual=output_query.numerical,
        expected=expected_query.numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_loaded_state_keeps_double_precision() -> None:
    # Distinct only in double precision, as after the Kumo recipe's cast.
    context = 1 + torch.arange(100, dtype=torch.float64).mul(1e-12)
    table = TableTensor.from_tensor(context[:, None])
    processor = RankGaussian().fit(table)
    loaded = RankGaussian()
    loaded.load_state_dict(processor.state_dict(), assign=True)
    torch.testing.assert_close(
        actual=loaded.transform(table).numerical,
        expected=processor.transform(table).numerical,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
