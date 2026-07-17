import torch
from benchmark.processor_speed_of_light import (
    POWER_LAMBDA_ATOL,
    POWER_OUTPUT_ATOL,
    QUANTILE_OUTPUT_ATOL,
    _batched_power_fit,
    _categorical_lookup_transform,
    _categorical_observed_mask,
    _category_permutation_transform,
    _feature_batched_quantile_row_major,
    _newton_power_fit,
    _power_log_likelihood,
    _quantile_fit_preindexed,
    _sigma_clip_fit_transform,
    _standard_scale_fit_transform,
    _standard_scale_inverse,
    _vectorized_quantile_row_major,
    _vectorized_yeojohnson,
)
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    CategoricalAlign,
    CategoryShuffle,
    Power,
    Quantile,
    SigmaClip,
    StandardScale,
)
from sdm.processing.power import _yeojohnson_transform


def test_vectorized_yeojohnson_matches_feature_loop() -> None:
    inp = torch.tensor(
        [
            [-2.0, -2.0, -2.0, -2.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [torch.nan, 2.0, 2.0, 2.0],
        ]
    )
    lambdas = torch.tensor([0.0, 2.0, 0.5, 1.5])
    expected = torch.stack(
        [
            _yeojohnson_transform(inp[:, index], float(lmbda))
            for index, lmbda in enumerate(lambdas)
        ],
        dim=1,
    )

    actual = _vectorized_yeojohnson(inp, lambdas)

    assert torch.allclose(actual, expected, equal_nan=True)


def test_batched_power_candidates_match_current_float32_output() -> None:
    generator = torch.Generator().manual_seed(0)
    inp = torch.randn(1_024, 4, generator=generator)
    inp[:, 0] = 1
    inp[::97, 1] = torch.nan
    inp[:, 2] = inp[:, 2].exp()
    table = TableTensor.from_tensor(inp)
    current = Power().fit(table)
    expected = current.transform(table).numerical

    states = (
        _batched_power_fit(
            inp,
            _power_log_likelihood,
            _vectorized_yeojohnson,
        ),
        _newton_power_fit(inp),
    )
    for lambdas, mean, scale in states:
        actual = (_vectorized_yeojohnson(inp, lambdas) - mean) / scale

        assert torch.allclose(
            lambdas,
            current.lambdas,
            rtol=0,
            atol=POWER_LAMBDA_ATOL,
        )
        assert torch.allclose(
            actual,
            expected,
            rtol=1e-3,
            atol=POWER_OUTPUT_ATOL,
            equal_nan=True,
        )


def test_batched_quantile_candidates_preserve_duplicates_and_nans() -> None:
    inp = torch.tensor(
        [
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [torch.nan, 1.0, 2.0],
            [2.0, 2.0, 3.0],
            [2.0, 3.0, torch.nan],
        ]
    )
    table = TableTensor.from_tensor(inp)
    processor = Quantile(
        n_quantiles=5,
        subsample=None,
        output_distribution="normal",
    ).fit(table)
    expected = processor.transform(table).numerical

    actual = _vectorized_quantile_row_major(
        inp,
        processor.quantiles,
        processor.references,
    )
    feature_batched = _feature_batched_quantile_row_major(
        inp,
        processor.quantiles,
        processor.references,
        batch_size=2,
    )

    assert torch.allclose(
        actual,
        expected,
        rtol=1e-6,
        atol=QUANTILE_OUTPUT_ATOL,
        equal_nan=True,
    )
    assert torch.allclose(
        feature_batched,
        actual,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_direct_scale_and_sigma_candidates_match_processors() -> None:
    inp = torch.tensor(
        [
            [1.0, 1.0, torch.nan],
            [1.0, 2.0, 3.0],
            [1.0, 20.0, 4.0],
            [1.0, 4.0, 5.0],
        ]
    )
    table = TableTensor.from_tensor(inp)

    scale_expected = StandardScale(epsilon=1e-6).fit_transform(table).numerical
    scale_actual, mean, scale = _standard_scale_fit_transform(
        inp,
        epsilon=1e-6,
    )
    inverse_actual = _standard_scale_inverse(scale_actual, mean, scale)
    inverse_expected = (
        StandardScale(epsilon=1e-6)
        .fit(table)
        .inverse_transform(TableTensor.from_tensor(scale_actual))
        .numerical
    )

    assert torch.allclose(
        scale_actual,
        scale_expected,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
    assert torch.allclose(inverse_actual, inverse_expected, equal_nan=True)

    sigma_expected = SigmaClip(threshold=4).fit_transform(table).numerical
    sigma_actual = _sigma_clip_fit_transform(inp, threshold=4)

    assert torch.allclose(
        sigma_actual,
        sigma_expected,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_direct_categorical_candidates_match_processors() -> None:
    fit_codes = torch.tensor(
        [
            [0, 2],
            [1, -1],
            [2, 0],
            [1, 2],
        ],
        dtype=torch.int32,
    )
    transform_codes = torch.tensor(
        [
            [0, 2],
            [3, 1],
            [-1, -1],
        ],
        dtype=torch.int32,
    )
    categories = (torch.arange(4), torch.arange(4))
    fit_table = TableTensor(
        columns={"categorical": ("a", "b")},
        categorical=CategoricalTensor(fit_codes, categories=categories),
    )
    transform_table = TableTensor(
        columns={"categorical": ("a", "b")},
        categorical=CategoricalTensor(transform_codes, categories=categories),
    )
    processor = CategoricalAlign(order="sorted").fit(fit_table)

    observed = _categorical_observed_mask(fit_codes, category_count=4)
    expected_observed = torch.tensor(
        [
            [True, True, True, False],
            [True, False, True, False],
        ]
    )
    assert torch.equal(observed.cpu(), expected_observed)

    mappings = torch.stack(
        [
            CategoricalAlign._category_mapping(
                actual=actual,
                expected=expected,
                device=torch.device("cpu"),
                column=f"c{index}",
            )
            for index, (actual, expected) in enumerate(
                zip(categories, processor._categories, strict=True)
            )
        ]
    )
    actual = _categorical_lookup_transform(transform_codes, mappings)
    expected = processor.transform(transform_table).categorical.as_tensor()

    assert torch.equal(actual, expected)


def test_direct_category_permutation_matches_shift_shuffle() -> None:
    codes = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    table = TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            codes,
            categories=(StringTensor.from_list(["negative", "positive"]),),
        ),
    )
    torch.manual_seed(0)
    processor = CategoryShuffle(method="shift").fit(table)
    expected = processor.transform(table).categorical.as_tensor()

    actual = _category_permutation_transform(codes, processor.permutations)

    assert torch.equal(actual, expected)


def test_quantile_fit_preindexed_matches_seeded_fit() -> None:
    inp = torch.arange(40, dtype=torch.float32).view(20, 2)
    references = torch.linspace(0, 1, 4)
    indices = torch.tensor([0, 3, 7, 9, 11, 15, 18, 19])

    expected = torch.nanquantile(inp[indices], references, dim=0)
    actual = _quantile_fit_preindexed(inp, references, indices)

    assert torch.equal(actual, expected)
