from typing import cast

import pytest
import torch

from sdm.models.tabfm.block import TabFMTransformerBlock
from sdm.models.tabfm.icl import ICLearning


def _module(
    is_classifier: bool,
    *,
    max_classes: int = 3,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ICLearning:
    return ICLearning(
        channels=4,
        num_blocks=1,
        num_heads=2,
        feedforward_channels=6,
        decoder_hidden_channels=5,
        is_classifier=is_classifier,
        max_classes=max_classes,
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("is_classifier", [True, False])
def test_icl_isolates_queries_and_uses_context(
    is_classifier: bool,
) -> None:
    module = _module(is_classifier, max_classes=10)
    block = cast(TabFMTransformerBlock, module.tf_icl.blocks[0])
    with torch.no_grad():
        torch.nn.init.eye_(block.attn.out_lin.weight)
    representations = torch.arange(48, dtype=torch.float32).view(3, 4, 4) / 20
    context_size = torch.tensor([0, 2, 4])
    targets = (
        torch.tensor([[0, 1, 2, 0], [0, 2, -100, 99], [2, 1, 0, 2]])
        if is_classifier
        else torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0],
                [0.5, -0.5, -100.0, 99.0],
                [2.0, 1.0, 0.0, -1.0],
            ]
        )
    )
    rows = torch.arange(4)[None, :]
    query = rows >= context_size[:, None]
    sentinel_values = (
        [-100, 99, -7, 4]
        if is_classifier
        else [float("nan"), float("inf"), -float("inf"), float("nan")]
    )
    sentinels = torch.tensor(
        [sentinel_values],
        dtype=targets.dtype,
    ).expand_as(targets)

    expected = module(representations, targets, context_size)
    changed_targets = torch.where(query, sentinels, targets)
    torch.testing.assert_close(
        module(representations, changed_targets, context_size),
        expected,
        rtol=0,
        atol=0,
    )

    changed_representations = representations.clone()
    changed_representations[1, 2] += torch.tensor([2.0, -1.0, 0.5, 1.0])
    changed = module(changed_representations, targets, context_size)
    torch.testing.assert_close(
        changed[1, [0, 1, 3]],
        expected[1, [0, 1, 3]],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(changed[1, 2], expected[1, 2])

    changed_targets = targets.clone()
    changed_targets[1, 0] = 1 if is_classifier else 2.5
    changed = module(representations, changed_targets, context_size)
    assert not torch.allclose(changed[1, 2:], expected[1, 2:])
    assert expected.shape == (3, 4, 10 if is_classifier else 1)


def test_icl_rejects_empty_classification_head() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        _module(True, max_classes=0)
