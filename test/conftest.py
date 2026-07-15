import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _torch_warn_always() -> None:
    torch.set_warn_always(True)
