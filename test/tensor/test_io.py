import pytest
import torch
from sdm.tensor.io import to_cudf
from sdm.testing import onlyCUDA


@onlyCUDA
def test_to_cudf() -> None:
    pytest.importorskip("cudf")
    tensor = torch.arange(6, device="cuda").view(2, 3)

    series = to_cudf(tensor)
    output = torch.as_tensor(series)

    assert output.equal(tensor.view(-1))
    assert output.data_ptr() == tensor.data_ptr()


@onlyCUDA
def test_to_cudf_noncontiguous() -> None:
    pytest.importorskip("cudf")
    tensor = torch.arange(12, device="cuda").view(3, 4).t()
    assert not tensor.is_contiguous()

    series = to_cudf(tensor)

    assert torch.as_tensor(series).equal(tensor.contiguous().view(-1))


def test_to_cudf_requires_cuda() -> None:
    with pytest.raises(ValueError, match="on a CUDA device"):
        to_cudf(torch.arange(3))
