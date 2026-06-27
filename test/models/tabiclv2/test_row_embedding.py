import pytest
import torch
from sdm import TaskType
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "task_type",
    [TaskType.classification, TaskType.regression],
)
def test_row_embedding_forward(
    device: torch.device,
    task_type: TaskType,
) -> None:
    batch_size, num_rows, num_cols, num_train = 2, 8, 6, 5
    channels, num_readout_tokens = 16, 2
    module = RowEmbedding(
        task_type=task_type,
        channels=channels,
        num_layers=2,
        num_heads=4,
        group_size=3,
        num_inducing_points=5,
        num_readout_tokens=num_readout_tokens,
        device=device,
    )
    x = torch.randn(batch_size, num_rows, num_cols, device=device)
    if task_type == TaskType.classification:
        y = torch.randint(0, 10, (batch_size, num_train), device=device)
    else:
        y = torch.randn(batch_size, num_train, device=device)

    out = module(x, y)
    assert out.shape == (batch_size, num_rows, num_readout_tokens * channels)
    assert out.dtype == x.dtype
    assert out.device == x.device
