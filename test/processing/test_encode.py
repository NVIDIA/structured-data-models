from __future__ import annotations

import pytest
import torch
from schemafm.processing import LabelEncode


def test_label_encode_1d_target_and_inverse() -> None:
    train = torch.tensor([2.0, 1.0, 2.0, 3.0])
    test = torch.tensor([3.0, 1.0, 2.0])

    processor = LabelEncode().fit(train)
    encoded = processor.transform(test)
    inverse = processor.inverse_transform(torch.tensor([0, 1, 2, -1]))

    assert torch.equal(processor.classes, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(encoded, torch.tensor([2, 0, 1]))
    assert torch.equal(
        torch.isnan(inverse), torch.tensor([False, False, False, True])
    )
    assert torch.equal(inverse[:3], torch.tensor([1.0, 2.0, 3.0]))


def test_label_encode_rejects_missing_and_unknown_labels() -> None:
    processor = LabelEncode().fit(torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError, match="unknown or missing"):
        processor.transform(torch.tensor([1.0, 3.0]))
    with pytest.raises(ValueError, match="unknown or missing"):
        processor.transform(torch.tensor([1.0, torch.nan]))


def test_label_encode_rejects_missing_labels_at_fit() -> None:
    with pytest.raises(ValueError, match="missing labels"):
        LabelEncode().fit(torch.tensor([1.0, torch.nan]))


def test_label_encode_inverse_handles_bool_labels() -> None:
    processor = LabelEncode().fit(torch.tensor([False, True]))

    assert torch.equal(
        processor.inverse_transform(torch.tensor([0, 1])),
        torch.tensor([False, True]),
    )
    with pytest.raises(ValueError, match="bool categories"):
        processor.inverse_transform(torch.tensor([0, -1]))
