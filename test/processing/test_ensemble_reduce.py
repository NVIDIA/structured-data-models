import pytest
import torch
from sdm import TableTensor
from sdm.processing import EnsembleReduce
from sdm.testing import withCUDA


@withCUDA
def test_ensemble_reduce_mean(device: torch.device) -> None:
    values = torch.arange(
        2 * 3 * 4 * 2,
        dtype=torch.float32,
        device=device,
    ).reshape(2, 3, 4, 2)
    table = TableTensor.from_tensor(values, columns=("a", "b"))
    processor = EnsembleReduce(method="mean")

    output = processor.transform(table)

    assert output.size() == (3, 4, 2)
    assert output.schema == table.schema
    assert output.device == device
    assert output.dtype == values.dtype
    torch.testing.assert_close(output.numerical, values.mean(dim=0))
    assert repr(processor) == "EnsembleReduce(method='mean')"


def test_ensemble_reduce_rejects_missing_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.ones(4, 2))

    with pytest.raises(ValueError, match="leading ensemble dimension"):
        EnsembleReduce().transform(table)


def test_ensemble_reduce_rejects_empty_ensemble_dimension() -> None:
    table = TableTensor.from_tensor(torch.empty(0, 4, 2))

    with pytest.raises(ValueError, match="at least one ensemble member"):
        EnsembleReduce().transform(table)


def test_ensemble_reduce_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be 'mean'"):
        EnsembleReduce(method="median")  # type: ignore
