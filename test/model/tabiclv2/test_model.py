import pytest
import torch
from sdm import TaskType
from sdm.model import TabICLv2


def make_module(
    task_type: TaskType,
    channels: int = 16,
    num_embedding_layers: int = 2,
    num_icl_layers: int = 2,
    num_heads: int = 4,
    group_size: int = 3,
    num_inducing_points: int = 5,
    num_readout_tokens: int = 2,
) -> TabICLv2:
    return TabICLv2(
        task_type=task_type,
        channels=channels,
        num_embedding_layers=num_embedding_layers,
        num_icl_layers=num_icl_layers,
        num_heads=num_heads,
        group_size=group_size,
        num_inducing_points=num_inducing_points,
        num_readout_tokens=num_readout_tokens,
    )


@pytest.mark.parametrize(
    "task_type",
    [TaskType.classification, TaskType.regression],
)
def test_tabiclv2_forward(task_type: TaskType) -> None:
    batch_size, num_rows, num_cols, num_train = 2, 8, 6, 5
    num_test = num_rows - num_train
    module = make_module(task_type)
    x = torch.randn(batch_size, num_rows, num_cols)
    if task_type == TaskType.classification:
        y = torch.randint(0, 10, (batch_size, num_train))
    else:
        y = torch.randn(batch_size, num_train)

    out = module(x, y)
    out_channels = 10 if task_type == TaskType.classification else 999
    assert out.shape == (batch_size, num_test, out_channels)
    assert out.dtype == x.dtype
    assert out.device == x.device
