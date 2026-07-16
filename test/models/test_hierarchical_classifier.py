import pytest
import torch
from sdm.models.tabiclv2.hierarchical_classifier import (
    HierarchicalClassifier,
)
from sdm.testing import withCUDA


@withCUDA
def test_hierarchical_classifier_grouping(device: torch.device) -> None:
    classifier = HierarchicalClassifier(num_classes=10)

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
    ("total_num_classes", "num_classes", "expected_calls"),
    [(3, 10, 1), (3, 2, 2), (25, 10, 4), (101, 10, 13)],
)
def test_hierarchical_classifier_probabilities(
    device: torch.device,
    total_num_classes: int,
    num_classes: int,
    expected_calls: int,
) -> None:
    calls = 0

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, num_classes))

    test_size = 2
    row_embeddings = torch.randn(
        total_num_classes + test_size,
        4,
        device=device,
    )
    y = torch.arange(total_num_classes, device=device)
    classifier = HierarchicalClassifier(num_classes=num_classes)

    probabilities = classifier(
        row_embeddings,
        y,
        num_classes=total_num_classes,
        predictor=predictor,
    )

    assert probabilities.size() == (test_size, total_num_classes)
    assert probabilities.dtype == row_embeddings.dtype
    assert probabilities.device == device
    assert calls == expected_calls
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(test_size, device=device),
    )

    expected = torch.full(
        (total_num_classes,),
        1 / total_num_classes,
        device=device,
    )
    if (total_num_classes, num_classes) == (3, 2):
        expected = torch.tensor([0.25, 0.25, 0.5], device=device)
    elif total_num_classes == 25:
        expected[:9] = 1 / 27
        expected[9:] = 1 / 24
    elif total_num_classes == 101:
        expected[:6] = 1 / 120
        expected[6:] = 1 / 100
    torch.testing.assert_close(probabilities[0], expected)


@withCUDA
def test_hierarchical_classifier_nonuniform_probabilities(
    device: torch.device,
) -> None:
    classifier = HierarchicalClassifier(num_classes=2)

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


@withCUDA
def test_hierarchical_classifier_batched_absent_classes(
    device: torch.device,
) -> None:
    classifier = HierarchicalClassifier(num_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique(sorted=True)
        torch.testing.assert_close(
            classes,
            torch.arange(classes.numel(), device=labels.device),
        )
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, 2))

    row_embeddings = torch.randn(2, 5, 4, device=device)
    y = torch.tensor([[1, 3, 4], [0, 2, 5]], device=device)

    probabilities = classifier(
        row_embeddings,
        y,
        num_classes=6,
        predictor=predictor,
    )

    expected = torch.tensor(
        [
            [0.0, 0.25, 0.0, 0.25, 0.5, 0.0],
            [0.25, 0.0, 0.25, 0.0, 0.0, 0.5],
        ],
        device=device,
    )
    torch.testing.assert_close(
        probabilities,
        expected.unsqueeze(1).expand(-1, 2, -1),
    )


def test_hierarchical_classifier_preserves_gradients() -> None:
    classifier = HierarchicalClassifier(
        num_classes=2,
        temperature=1.0,
    )
    logits = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return logits + rows[labels.size(0) :, :2]

    row_embeddings = torch.randn(5, 4, requires_grad=True)
    probabilities = classifier(
        row_embeddings,
        torch.arange(3),
        num_classes=3,
        predictor=predictor,
    )
    probabilities[:, 0].sum().backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert row_embeddings.grad is not None
    assert row_embeddings.grad.abs().sum() > 0


def test_hierarchical_classifier_single_class_preserves_gradients() -> None:
    classifier = HierarchicalClassifier(num_classes=2)
    row_embeddings = torch.randn(3, 4, requires_grad=True)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise AssertionError("A single-class context needs no predictor call")

    probabilities = classifier(
        row_embeddings,
        torch.zeros(2, dtype=torch.long),
        num_classes=1,
        predictor=predictor,
    )
    probabilities.sum().backward()

    torch.testing.assert_close(probabilities, torch.ones(1, 1))
    assert row_embeddings.grad is not None


def test_hierarchical_classifier_empty_batch() -> None:
    classifier = HierarchicalClassifier(num_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise AssertionError("The predictor must not run for an empty batch")

    row_embeddings = torch.randn(0, 4, 3, requires_grad=True)
    probabilities = classifier(
        row_embeddings,
        torch.zeros(0, 2, dtype=torch.long),
        num_classes=5,
        predictor=predictor,
    )

    assert probabilities.size() == (0, 2, 5)
    assert probabilities.dtype == torch.float32
    probabilities.sum().backward()
    assert row_embeddings.grad is not None


def test_hierarchical_classifier_rejects_invalid_num_classes() -> None:
    with pytest.raises(ValueError, match="'num_classes' to be at least two"):
        HierarchicalClassifier(num_classes=1)
