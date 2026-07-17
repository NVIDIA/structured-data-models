import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session", autouse=True)
def _torch_warn_always() -> None:
    torch.set_warn_always(True)
