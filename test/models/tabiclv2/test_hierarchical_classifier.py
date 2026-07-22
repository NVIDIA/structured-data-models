import pytest
import torch
from sdm.models.tabiclv2.hierarchical_classifier import (
    HierarchicalClassifier,
)
from sdm.testing import withCUDA


def test_hierarchical_classifier_grouping() -> None:
    classifier = HierarchicalClassifier(num_classes=10)
    device = torch.device("cpu")

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
    (
        "total_num_classes",
        "num_classes",
        "expected_calls",
        "expected_probabilities",
    ),
    [
        pytest.param(
            3,
            10,
            1,
            [1 / 3, 1 / 3, 1 / 3],
            id="no-hierarchy",
        ),
        pytest.param(
            5,
            2,
            4,
            [1 / 8, 1 / 8, 1 / 4, 1 / 4, 1 / 4],
            id="multi-level-with-singleton",
        ),
    ],
)
def test_hierarchical_classifier_log_probs(
    device: torch.device,
    total_num_classes: int,
    num_classes: int,
    expected_calls: int,
    expected_probabilities: list[float],
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

    log_probs = classifier(
        row_embeddings,
        y,
        num_classes=total_num_classes,
        predictor=predictor,
    )

    assert calls == expected_calls
    torch.testing.assert_close(
        log_probs,
        row_embeddings.new_tensor(expected_probabilities)
        .log()
        .expand(test_size, -1),
    )


def test_hierarchical_classifier_nonuniform_log_probs() -> None:
    classifier = HierarchicalClassifier(num_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        test_size = rows.size(0) - labels.size(0)
        if labels.size(0) == 3:
            probabilities = rows.new_tensor([0.6, 0.4])
        else:
            probabilities = rows.new_tensor([0.25, 0.75])
        return probabilities.log().expand(test_size, -1)

    row_embeddings = torch.randn(5, 4)
    log_probs = classifier(
        row_embeddings,
        torch.arange(3, dtype=torch.int32),
        num_classes=3,
        predictor=predictor,
    )

    expected = torch.tensor([0.15, 0.45, 0.4]).log().expand(2, -1)
    torch.testing.assert_close(log_probs, expected)


def test_hierarchical_classifier_batched_absent_classes() -> None:
    classifier = HierarchicalClassifier(num_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        classes = labels.unique(sorted=True)
        torch.testing.assert_close(
            classes,
            torch.arange(classes.numel(), device=labels.device),
        )
        test_size = rows.size(0) - labels.size(0)
        return rows.new_zeros((test_size, 2))

    row_embeddings = torch.randn(2, 5, 4)
    y = torch.tensor([[1, 3, 4], [0, 2, 5]])

    log_probs = classifier(
        row_embeddings,
        y,
        num_classes=6,
        predictor=predictor,
    )

    expected_probabilities = torch.tensor(
        [
            [0.0, 0.25, 0.0, 0.25, 0.5, 0.0],
            [0.25, 0.0, 0.25, 0.0, 0.0, 0.5],
        ],
    )
    torch.testing.assert_close(
        log_probs,
        expected_probabilities.log().unsqueeze(1).expand(-1, 2, -1),
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
    log_probs = classifier(
        row_embeddings,
        torch.arange(3),
        num_classes=3,
        predictor=predictor,
    )
    log_probs[:, 0].sum().backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
    assert row_embeddings.grad is not None
    assert row_embeddings.grad.abs().sum() > 0


def test_hierarchical_classifier_empty_batch() -> None:
    classifier = HierarchicalClassifier(num_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise AssertionError("The predictor must not run for an empty batch")

    row_embeddings = torch.randn(0, 4, 3)
    log_probs = classifier(
        row_embeddings,
        torch.zeros(0, 2, dtype=torch.long),
        num_classes=5,
        predictor=predictor,
    )

    assert log_probs.size() == (0, 2, 5)
    assert log_probs.dtype == torch.float32


def test_hierarchical_classifier_rejects_invalid_num_classes() -> None:
    with pytest.raises(ValueError, match="'num_classes' to be at least two"):
        HierarchicalClassifier(num_classes=1)
