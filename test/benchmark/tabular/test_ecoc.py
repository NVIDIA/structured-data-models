import hashlib

import numpy as np
import pytest

from benchmark.tabular._ecoc import (
    align_symbol_probabilities,
    decode_many_class_probabilities,
    many_class_codebook,
    many_class_estimator_count,
)


@pytest.mark.parametrize(
    ("n_classes", "expected"),
    [(11, 8), (24, 8), (81, 9)],
)
def test_many_class_estimator_count(
    n_classes: int,
    expected: int,
) -> None:
    assert many_class_estimator_count(n_classes, 10) == expected


def test_many_class_estimator_count_requires_coverage() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        many_class_estimator_count(24, 10, requested=2)


def test_many_class_codebook_is_deterministic_and_complete() -> None:
    codebooks = [
        many_class_codebook(24, 10, 8, random_state=7) for _ in range(2)
    ]

    np.testing.assert_array_equal(codebooks[0], codebooks[1])
    assert codebooks[0].shape == (8, 24)
    assert codebooks[0].min() == 0
    assert codebooks[0].max() == 9
    assert np.all((codebooks[0] != 9).sum(axis=0) > 0)
    digest = hashlib.sha256(codebooks[0].tobytes()).hexdigest()
    assert (
        digest == "5ecb75ae2c5f31baad56e370bd81d644"
        "19728f9c086463ee64fe2cd61c780729"
    )


def _decode_oracle(
    probabilities: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    rows, queries, _ = probabilities.shape
    classes = codebook.shape[1]
    rest = probabilities.shape[2] - 1
    scores = np.zeros((queries, classes))
    for query in range(queries):
        for cls in range(classes):
            values = [
                np.log(
                    max(probabilities[row, query, codebook[row, cls]], 1e-12)
                )
                for row in range(rows)
                if codebook[row, cls] != rest
            ]
            scores[query, cls] = np.mean(values)
    scores -= scores.max(axis=1, keepdims=True)
    result = np.exp(scores)
    return result / result.sum(axis=1, keepdims=True)


def test_decode_many_class_probabilities_matches_oracle() -> None:
    rng = np.random.RandomState(4)
    codebook = many_class_codebook(11, 10, 8, random_state=7)
    probabilities = rng.dirichlet(
        np.ones(10),
        size=(8, 5),
    )

    actual = decode_many_class_probabilities(probabilities, codebook)
    expected = _decode_oracle(probabilities, codebook)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)
    assert np.isfinite(actual).all()
    assert (actual >= 0).all()


def test_decode_masks_the_rest_symbol() -> None:
    codebook = np.array([[0, 2], [2, 0]])
    probabilities = np.array(
        [
            [[0.8, 0.1, 0.1]],
            [[0.7, 0.1, 0.2]],
        ]
    )
    changed_rest = probabilities.copy()
    changed_rest[0, 0] = [0.8, 0.19, 0.01]
    changed_rest[1, 0] = [0.7, 0.29, 0.01]

    actual = decode_many_class_probabilities(probabilities, codebook)
    changed = decode_many_class_probabilities(changed_rest, codebook)

    np.testing.assert_allclose(actual, changed)


def test_align_symbol_probabilities_inserts_zeros() -> None:
    probabilities = np.array([[0.25, 0.75], [0.6, 0.4]])

    actual = align_symbol_probabilities(
        probabilities,
        symbols=np.array([1, 3]),
        alphabet_size=5,
    )

    expected = np.array(
        [
            [0.0, 0.25, 0.0, 0.75, 0.0],
            [0.0, 0.6, 0.0, 0.4, 0.0],
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_base_estimators_must_average_before_decoding() -> None:
    codebook = np.array([[0, 1], [1, 0]])
    member_probabilities = np.array(
        [
            [
                [[0.99, 0.01, 0.0]],
                [[0.20, 0.80, 0.0]],
            ],
            [
                [[0.80, 0.20, 0.0]],
                [[0.01, 0.99, 0.0]],
            ],
        ]
    )

    expected = decode_many_class_probabilities(
        member_probabilities.mean(axis=1),
        codebook,
    )
    wrong = np.mean(
        [
            decode_many_class_probabilities(
                member_probabilities[:, member],
                codebook,
            )
            for member in range(member_probabilities.shape[1])
        ],
        axis=0,
    )

    assert not np.allclose(expected, wrong)
