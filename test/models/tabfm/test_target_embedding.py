import pytest
import torch

from sdm.models.tabfm.target_embedding import TargetEmbedding


@pytest.mark.parametrize("is_classifier", [True, False])
def test_target_embedding(
    is_classifier: bool,
) -> None:
    module = TargetEmbedding(channels=3, is_classifier=is_classifier)
    targets = (
        torch.tensor([[0, 1, 2], [2, 1, 0]])
        if is_classifier
        else torch.tensor([[0.25, -0.5, 1.0], [2.0, -1.0, 0.5]])
    )
    context_size = torch.tensor([2, 1])
    query = torch.arange(3).unsqueeze(0) >= context_size.unsqueeze(-1)

    output = module(targets, context_size)
    assert output.shape == (*targets.shape, 3)
    assert torch.count_nonzero(output.masked_select(query.unsqueeze(-1))) == 0

    changed_query = targets.masked_fill(
        query, 9 if is_classifier else float("nan")
    )
    torch.testing.assert_close(module(changed_query, context_size), output)

    changed_context = targets.clone()
    changed_context[0, 0] = 1 if is_classifier else 3.0
    changed_output = module(changed_context, context_size)
    assert not torch.equal(changed_output[0, 0], output[0, 0])

    unchanged = torch.ones_like(targets, dtype=torch.bool)
    unchanged[0, 0] = False
    torch.testing.assert_close(
        changed_output.masked_select(unchanged.unsqueeze(-1)),
        output.masked_select(unchanged.unsqueeze(-1)),
    )
