import pytest
import torch
from sdm.nn import HierarchicalClassifier
from sdm.testing import withCUDA


@withCUDA
def test_hierarchical_classifier_grouping(device: torch.device) -> None:
    classifier = HierarchicalClassifier(max_classes=10)

    assignments, num_groups = classifier._grouping(
        num_classes=5,
        device=device,
    )
    assert num_groups == 1
    torch.testing.assert_close(
        assignments,
        torch.zeros(5, dtype=torch.long, device=device),
    )

    assignments, num_groups = classifier._grouping(
        num_classes=25,
        device=device,
    )
    assert num_groups == 3
    torch.testing.assert_close(
        assignments,
        torch.tensor([0] * 9 + [1] * 8 + [2] * 8, device=device),
    )

    assignments, num_groups = classifier._grouping(
        num_classes=101,
        device=device,
    )
    assert num_groups == 10
    torch.testing.assert_close(
        assignments,
        torch.arange(10, device=device).repeat_interleave(
            torch.tensor([11] + [10] * 9, device=device)
        ),
    )


@withCUDA
@pytest.mark.parametrize(
    ("num_classes", "max_classes", "expected_calls"),
    [(3, 10, 1), (3, 2, 2), (25, 10, 4), (101, 10, 13)],
)
def test_hierarchical_classifier_probabilities(
    device: torch.device,
    num_classes: int,
    max_classes: int,
    expected_calls: int,
) -> None:
    calls = 0

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, max_classes))

    test_size = 2
    row_embeddings = torch.randn(
        num_classes + test_size,
        4,
        device=device,
    )
    y = torch.arange(num_classes, device=device)
    classifier = HierarchicalClassifier(max_classes=max_classes)

    probabilities = classifier(
        row_embeddings,
        y,
        num_classes=num_classes,
        predictor=predictor,
    )

    assert probabilities.size() == (test_size, num_classes)
    assert probabilities.dtype == row_embeddings.dtype
    assert probabilities.device == device
    assert calls == expected_calls
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(test_size, device=device),
    )

    expected = torch.full(
        (num_classes,),
        1 / num_classes,
        device=device,
    )
    if (num_classes, max_classes) == (3, 2):
        expected = torch.tensor([0.25, 0.25, 0.5], device=device)
    elif num_classes == 25:
        expected[:9] = 1 / 27
        expected[9:] = 1 / 24
    elif num_classes == 101:
        expected[:6] = 1 / 120
        expected[6:] = 1 / 100
    torch.testing.assert_close(probabilities[0], expected)


@withCUDA
def test_hierarchical_classifier_nonuniform_probabilities(
    device: torch.device,
) -> None:
    classifier = HierarchicalClassifier(
        max_classes=2,
        temperature=1.0,
    )

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        test_size = rows.size(0) - labels.size(0)
        if labels.size(0) == 3:
            probabilities = rows.new_tensor([0.6, 0.4])
        else:
            probabilities = rows.new_tensor([0.25, 0.75])
        return probabilities.log().expand(test_size, -1)

    row_embeddings = torch.randn(5, 4, device=device)
    probabilities = classifier(
        row_embeddings,
        torch.arange(3, device=device),
        num_classes=3,
        predictor=predictor,
    )

    expected = torch.tensor([0.15, 0.45, 0.4], device=device).expand(2, -1)
    torch.testing.assert_close(probabilities, expected)


def test_hierarchical_classifier_requires_two_native_classes() -> None:
    classifier = HierarchicalClassifier(max_classes=1)

    with pytest.raises(ValueError, match="at least two native classes"):
        classifier(
            torch.randn(3, 4),
            torch.arange(2),
            num_classes=2,
            predictor=lambda rows, labels: rows.new_zeros((1, 1)),
        )


@pytest.mark.parametrize(
    ("max_classes", "temperature", "match"),
    [
        (0, 0.9, "'max_classes' to be positive"),
        (2, 0.0, "'temperature' to be finite and positive"),
    ],
)
def test_hierarchical_classifier_rejects_invalid_init(
    max_classes: int,
    temperature: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        HierarchicalClassifier(
            max_classes=max_classes,
            temperature=temperature,
        )
