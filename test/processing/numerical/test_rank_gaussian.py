# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import Tensor

from sdm import TableTensor
from sdm.processing import RankGaussian
from sdm.testing import onlyCUDA, withCUDA

# Enough knots to keep every fitted value.
ALL_KNOTS = 2**20


def assert_same(actual: Tensor, expected: Tensor) -> None:
    torch.testing.assert_close(
        actual, expected, rtol=0, atol=0, equal_nan=True
    )


def few_distinct_context(
    *shape: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    # [*shape, 5] columns with at most four distinct finite values: ties with
    # non-finite values, ties with signed zeros, two rare values between two
    # large ties, a constant and a missing column.
    context = torch.randint(4, (*shape, 5), device=device).to(dtype) * 2.5
    context[..., ::4, 0] = torch.nan
    context[..., 1::5, 0] = torch.inf
    context[..., 2::7, 0] = -torch.inf
    context[..., 1] = -context[..., 1]
    rows = torch.arange(shape[-1], device=device)
    context[..., 2] = rows.ge(shape[-1] // 2).to(dtype) * 7.5
    context[..., 1:2, 2] = 2.5
    context[..., 2:3, 2] = 5.0
    context[..., 3] = 3.0
    context[..., 4] = torch.nan
    return context


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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mid_ranks_and_interpolated_query(
    dtype: torch.dtype, device: torch.device
) -> None:
    context = torch.tensor(
        [[0.0], [1.0], [1.0], [2.0]], dtype=dtype, device=device
    )
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    probabilities = context.new_tensor([[0.125], [0.5], [0.5], [0.875]])
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(context)).numerical,
        torch.special.ndtri(probabilities),
    )
    query = context.new_tensor([[-100.0], [0.5], [1.5], [100.0], [torch.nan]])
    expected = context.new_tensor(
        [[0.125], [0.3125], [0.6875], [0.875], [torch.nan]]
    )
    torch.testing.assert_close(
        processor.transform(TableTensor.from_tensor(query)).numerical,
        torch.special.ndtri(expected),
        equal_nan=True,
    )


@withCUDA
def test_ties_at_endpoints_use_mid_ranks(device: torch.device) -> None:
    context = torch.tensor([[0.0], [0.0], [2.0], [2.0]], device=device)
    output = RankGaussian().fit_transform(TableTensor.from_tensor(context))
    probabilities = context.new_tensor([[0.25], [0.25], [0.75], [0.75]])
    torch.testing.assert_close(
        output.numerical, torch.special.ndtri(probabilities)
    )


@withCUDA
@pytest.mark.parametrize("n_rows", [1, 4])
def test_constants_and_missing_columns(
    n_rows: int, device: torch.device
) -> None:
    context = torch.tensor([[7.0, torch.nan]], device=device).expand(
        n_rows, -1
    )
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
def test_nonfinite_context_does_not_change_observed_ranks(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[0.0], [1.0], [torch.nan], [1.0], [torch.inf], [-torch.inf], [2.0]],
        device=device,
    )
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    reference = RankGaussian().fit(
        TableTensor.from_tensor(context[[0, 1, 3, 6]])
    )
    query = TableTensor.from_tensor(context)
    output = processor.transform(query).numerical
    torch.testing.assert_close(
        output, reference.transform(query).numerical, equal_nan=True
    )
    assert torch.equal(output.isnan(), context.isnan())
    assert output[~context.isnan()].isfinite().all()


@withCUDA
def test_query_batch_does_not_change_fitted_ranks(
    device: torch.device,
) -> None:
    context = torch.arange(10.0, device=device).unsqueeze(-1)
    processor = RankGaussian().fit(TableTensor.from_tensor(context))
    query = context.new_tensor([[1.5], [torch.nan], [5.5]])
    expected = processor.transform(TableTensor.from_tensor(query)).numerical
    extended = torch.cat([query, context.new_tensor([[-1e12], [1e12]])])
    actual = processor.transform(TableTensor.from_tensor(extended)).numerical
    torch.testing.assert_close(actual[:3], expected, equal_nan=True)


@withCUDA
@pytest.mark.parametrize("max_knots", [4, 8192])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_batched_missing_values_match_independent_columns(
    batch_shape: tuple[int, ...], max_knots: int, device: torch.device
) -> None:
    context = torch.randn(*batch_shape, 17, 4, device=device)
    context[..., 1, 0] = torch.nan
    context[..., :3, 1] = torch.nan
    context[..., :, 2] = 2.0
    context[..., :, 3] = torch.nan
    query = torch.randn(*batch_shape, 7, 4, device=device)
    query[..., 0, 0] = torch.nan
    processor = RankGaussian(max_knots=max_knots).fit(
        TableTensor.from_tensor(context)
    )
    output = processor.transform(TableTensor.from_tensor(query)).numerical
    for train, test, actual in zip(
        context.reshape(-1, 17, 4),
        query.reshape(-1, 7, 4),
        output.reshape(-1, 7, 4),
        strict=True,
    ):
        for column in range(4):
            reference = RankGaussian(max_knots=max_knots).fit(
                TableTensor.from_tensor(train[:, column : column + 1])
            )
            expected = reference.transform(
                TableTensor.from_tensor(test[:, column : column + 1])
            ).numerical
            torch.testing.assert_close(
                actual[:, column : column + 1], expected, equal_nan=True
            )


@pytest.mark.parametrize("max_knots", [0, 1])
def test_rejects_fewer_than_two_knots(max_knots: int) -> None:
    with pytest.raises(ValueError, match="max_knots"):
        RankGaussian(max_knots=max_knots)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1,), (30,), (3, 30), (2, 3, 30)])
def test_few_distinct_values_match_all_knots(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    context = few_distinct_context(*shape, dtype=dtype, device=device)
    table = TableTensor.from_tensor(context)
    query = TableTensor.from_tensor(edge_query(context))
    processor = RankGaussian(max_knots=4)
    reference = RankGaussian(max_knots=ALL_KNOTS)
    assert_same(
        actual=processor.fit_transform(table).numerical,
        expected=reference.fit_transform(table).numerical,
    )
    assert_same(
        actual=processor.transform(query).numerical,
        expected=reference.transform(query).numerical,
    )


@withCUDA
def test_strided_members_match_all_knots(device: torch.device) -> None:
    members = few_distinct_context(6, 30, dtype=torch.float32, device=device)
    context = members[::2]
    query = TableTensor.from_tensor(edge_query(context))
    processor = RankGaussian(max_knots=4)
    output = processor.fit_transform(TableTensor.from_tensor(context))
    reference = RankGaussian(max_knots=ALL_KNOTS)
    expected = reference.fit_transform(
        TableTensor.from_tensor(context.contiguous())
    )
    assert_same(actual=output.numerical, expected=expected.numerical)
    assert_same(
        actual=processor.transform(query).numerical,
        expected=reference.transform(query).numerical,
    )


@withCUDA
@pytest.mark.parametrize("max_knots", [2, 16, 64])
def test_many_distinct_values_stay_within_one_knot_spacing(
    max_knots: int, device: torch.device
) -> None:
    # Symmetric and skewed columns without ties, and a two-valued column. For
    # 300 values without ties, these knot counts bound the error by the knot
    # spacing whatever the values.
    context = torch.randn(2, 300, 3, dtype=torch.float64, device=device)
    context[..., 1] = context[..., 1].mul(2).exp()
    context[..., 2] = context[..., 2].sign()
    table = TableTensor.from_tensor(context)
    processor = RankGaussian(max_knots=max_knots).fit(table)
    reference = RankGaussian(max_knots=ALL_KNOTS).fit(table)
    output = processor.transform(table).numerical
    expected = reference.transform(table).numerical
    assert_same(actual=output[..., 2], expected=expected[..., 2])

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
    assert_same(
        actual=processor.transform(TableTensor.from_tensor(beyond)).numerical,
        expected=torch.cat([lower, upper] * 3, dim=-2),
    )

    values = context.sort(dim=-2).values
    midpoints = values[..., :-1, :].lerp(values[..., 1:, :], 0.5)
    queries = torch.cat([values, midpoints, beyond], dim=-2)
    scores = processor.transform(
        TableTensor.from_tensor(queries.sort(dim=-2).values)
    ).numerical
    assert scores.diff(dim=-2).ge(0).all()


@withCUDA
def test_nonfinite_context_does_not_change_coarse_knots(
    device: torch.device,
) -> None:
    finite = torch.randn(200, 2, dtype=torch.float64, device=device)
    nonfinite = finite.new_tensor([torch.nan, torch.inf, -torch.inf])
    context = torch.cat([finite, nonfinite[:, None].repeat(20, 2)])
    context = context[torch.randperm(context.size(0), device=device)]
    query = torch.cat([context, edge_query(context)])
    processor = RankGaussian(max_knots=16).fit(
        TableTensor.from_tensor(context)
    )
    reference = RankGaussian(max_knots=16).fit(TableTensor.from_tensor(finite))
    assert_same(
        actual=processor.transform(TableTensor.from_tensor(query)).numerical,
        expected=reference.transform(TableTensor.from_tensor(query)).numerical,
    )


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
    reference = RankGaussian(max_knots=ALL_KNOTS)
    expected = reference.fit_transform(table).numerical
    spacing = (expected.max() - expected.min()) / 63
    assert (output - expected).abs().le(spacing).all()


@onlyCUDA
@pytest.mark.parametrize("max_knots", [16, 8192])
def test_chunks_match_single_chunk(
    max_knots: int, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        processor = RankGaussian(max_knots=max_knots)
        expected = processor.fit_transform(table)
        expected_query = processor.transform(query)
        # Chunks of a single column in fit and a single row in transform:
        monkeypatch.setenv("SDM_CHUNK_MEMORY_FRACTION", "1e-12")
        processor = RankGaussian(max_knots=max_knots)
        output = processor.fit_transform(table)
        output_query = processor.transform(query)

    assert_same(actual=output.numerical, expected=expected.numerical)
    assert_same(
        actual=output_query.numerical, expected=expected_query.numerical
    )


@withCUDA
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_keeps_input_dtype(dtype: torch.dtype, device: torch.device) -> None:
    context = torch.randn(40, 3, device=device).to(dtype)
    context[::5, 0] = torch.nan
    context[:, 2] = context[:, 2].round()
    query = torch.cat(
        [torch.randn(8, 3, device=device).to(dtype) * 3, edge_query(context)]
    )
    processor = RankGaussian(max_knots=8)
    output = processor.fit_transform(TableTensor.from_tensor(context))
    transformed = processor.transform(TableTensor.from_tensor(query))
    assert output.numerical.dtype == dtype
    assert transformed.numerical.dtype == dtype

    reference = RankGaussian(max_knots=8).fit(
        TableTensor.from_tensor(context.double())
    )
    for inputs, actual in ((context, output), (query, transformed)):
        expected = reference.transform(
            TableTensor.from_tensor(inputs.double())
        )
        torch.testing.assert_close(
            actual=actual.numerical,
            expected=expected.numerical.to(dtype),
            equal_nan=True,
        )


@withCUDA
def test_empty_query_gives_empty_output(device: torch.device) -> None:
    context = TableTensor.from_tensor(torch.randn(10, 2, device=device))
    processor = RankGaussian(max_knots=4).fit(context)
    query = TableTensor.from_tensor(torch.empty(0, 2, device=device))
    assert processor.transform(query).numerical.shape == (0, 2)


@withCUDA
def test_context_without_rows_gives_missing_columns(
    device: torch.device,
) -> None:
    context = TableTensor.from_tensor(torch.empty(0, 2, device=device))
    processor = RankGaussian().fit(context)
    query = torch.tensor([[1.0, torch.nan]], device=device)
    output = processor.transform(TableTensor.from_tensor(query)).numerical
    assert output.isnan().all()


@withCUDA
def test_loaded_state_keeps_double_precision(device: torch.device) -> None:
    # Distinct only in double precision, as after the Kumo recipe's cast.
    context = 1 + torch.arange(100, dtype=torch.float64, device=device).mul(
        1e-12
    )
    table = TableTensor.from_tensor(context[:, None])
    processor = RankGaussian().fit(table)
    loaded = RankGaussian().to(device)
    loaded.load_state_dict(processor.state_dict())
    assert_same(
        actual=loaded.transform(table).numerical,
        expected=processor.transform(table).numerical,
    )
