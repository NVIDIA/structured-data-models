import pytest
import torch
from sdm import TaskType
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.testing import withCUDA
from torch import Tensor


@withCUDA
@pytest.mark.parametrize(
    "task_type",
    [TaskType.classification, TaskType.regression],
)
def test_icl_block(
    device: torch.device,
    task_type: TaskType,
) -> None:
    batch_size = 2
    num_train = 5
    num_test = 3
    channels = 16
    module = ICLBlock(
        task_type=task_type,
        channels=channels,
        num_layers=2,
        num_heads=2,
        device=device,
    )

    x = torch.randn(
        batch_size,
        num_train + num_test,
        channels,
        device=device,
    )
    if task_type == TaskType.classification:
        y: Tensor = torch.randint(
            0, 10, (batch_size, num_train), device=device
        )
        out_channels = 10
    else:
        y = torch.randn(batch_size, num_train, device=device)
        out_channels = 999

    out = module(x=x, y=y)
    assert out.shape == (batch_size, num_test, out_channels)
    assert out.dtype == x.dtype
    assert out.device == x.device
