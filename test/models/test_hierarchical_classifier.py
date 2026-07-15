import pytest
import torch
from sdm.models.tabiclv2.hierarchical_classifier import (
    HierarchicalClassifier,
)
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
    classifier = HierarchicalClassifier(max_classes=2)

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
    classifier = HierarchicalClassifier(max_classes=2)

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
        max_classes=2,
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
    classifier = HierarchicalClassifier(max_classes=2)
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


def test_hierarchical_classifier_validates_shapes() -> None:
    classifier = HierarchicalClassifier(max_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return rows.new_zeros((1, 2))

    with pytest.raises(ValueError, match="share batch dimensions"):
        classifier(
            torch.randn(2, 4, 3),
            torch.zeros(3, 2, dtype=torch.long),
            num_classes=2,
            predictor=predictor,
        )

    with pytest.raises(ValueError, match="no more labels"):
        classifier(
            torch.randn(2, 3),
            torch.zeros(3, dtype=torch.long),
            num_classes=2,
            predictor=predictor,
        )


def test_hierarchical_classifier_empty_batch() -> None:
    classifier = HierarchicalClassifier(max_classes=2)

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


@pytest.mark.parametrize("y", [torch.tensor([-1, 0]), torch.tensor([0, 2])])
def test_hierarchical_classifier_validates_labels(y: torch.Tensor) -> None:
    classifier = HierarchicalClassifier(max_classes=2)

    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        classifier(
            torch.randn(3, 4),
            y,
            num_classes=2,
            predictor=lambda rows, labels: rows.new_zeros((1, 2)),
        )


def test_hierarchical_classifier_requires_floating_embeddings() -> None:
    classifier = HierarchicalClassifier(max_classes=2)

    with pytest.raises(TypeError, match="floating-point row embeddings"):
        classifier(
            torch.ones(3, 4, dtype=torch.long),
            torch.arange(2),
            num_classes=2,
            predictor=lambda rows, labels: rows.new_zeros((1, 2)),
        )


@pytest.mark.parametrize(
    ("mode", "error", "match"),
    [
        ("rank", ValueError, "two dimensions"),
        ("rows", ValueError, "one logit row"),
        ("classes", ValueError, "at least one logit"),
        ("dtype", TypeError, "floating-point logits"),
    ],
)
def test_hierarchical_classifier_validates_predictor_output(
    mode: str,
    error: type[Exception],
    match: str,
) -> None:
    classifier = HierarchicalClassifier(max_classes=2)

    def predictor(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if mode == "rank":
            return rows.new_zeros((1, 1, 2))
        if mode == "rows":
            return rows.new_zeros((2, 2))
        if mode == "classes":
            return rows.new_zeros((1, 1))
        return torch.zeros((1, 2), dtype=torch.long, device=rows.device)

    with pytest.raises(error, match=match):
        classifier(
            torch.randn(3, 4),
            torch.arange(2),
            num_classes=2,
            predictor=predictor,
        )


@pytest.mark.parametrize(
    ("max_classes", "temperature", "match"),
    [
        (1, 0.9, "'max_classes' to be at least two"),
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
