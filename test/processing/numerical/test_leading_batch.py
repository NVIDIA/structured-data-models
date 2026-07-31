import copy

import pytest
import torch

from sdm import TableTensor
from sdm.processing import (
    ClipQuantiles,
    ClipSigma,
    ImputeMean,
    PowerTransform,
    Processor,
    Standardize,
)
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "processor",
    [
        Standardize(),
        ClipQuantiles(q_low=0.2, q_high=0.8),
        ClipSigma(threshold=1.5),
        PowerTransform(),
    ],
    ids=("standardize", "clip-quantiles", "clip-sigma", "power"),
)
def test_leading_batch_matches_independent_fits(
    device: torch.device,
    processor: Processor,
) -> None:
    fit_values = torch.tensor(
        [
            [
                [-4.0, 1.0],
                [-1.0, 2.0],
                [0.0, 4.0],
                [2.0, 8.0],
                [7.0, 16.0],
            ],
            [
                [-20.0, -3.0],
                [-10.0, -1.0],
                [0.0, 0.0],
                [10.0, 1.0],
                [20.0, 3.0],
            ],
        ],
        device=device,
    )
    query_values = torch.tensor(
        [
            [[-2.0, 3.0], [5.0, 12.0]],
            [[-15.0, -2.0], [15.0, 2.0]],
        ],
        device=device,
    )
    columns = ("left", "right")
    batched = copy.deepcopy(processor)

    actual_fit = batched.fit_transform(
        TableTensor.from_tensor(fit_values, columns=columns)
    )
    actual_query = batched.transform(
        TableTensor.from_tensor(query_values, columns=columns)
    )

    expected_fit = []
    expected_query = []
    for batch in range(fit_values.size(0)):
        scalar = copy.deepcopy(processor)
        scalar_fit = TableTensor.from_tensor(
            fit_values[batch],
            columns=columns,
        )
        scalar_query = TableTensor.from_tensor(
            query_values[batch],
            columns=columns,
        )
        expected_fit.append(scalar.fit_transform(scalar_fit).numerical)
        expected_query.append(scalar.transform(scalar_query).numerical)

    torch.testing.assert_close(
        actual_fit.numerical,
        torch.stack(expected_fit),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        actual_query.numerical,
        torch.stack(expected_query),
        rtol=2e-5,
        atol=2e-5,
    )


@withCUDA
def test_impute_mean_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    fit_values = torch.tensor(
        [
            [[1.0, float("nan")], [3.0, 4.0]],
            [[10.0, 20.0], [float("nan"), 40.0]],
        ],
        device=device,
    )
    query_values = torch.full(
        (2, 1, 2),
        float("nan"),
        device=device,
    )
    processor = ImputeMean().fit(TableTensor.from_tensor(fit_values))

    actual = processor.transform(TableTensor.from_tensor(query_values))

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor(
            [[[2.0, 4.0]], [[10.0, 30.0]]],
            device=device,
        ),
    )
