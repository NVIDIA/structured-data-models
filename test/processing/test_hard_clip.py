import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import HardClip


def test_hard_clip_clamps_fixed_bounds_and_preserves_metadata() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[-101.0, -100.0], [100.0, 101.0]]),
        columns=("a", "b"),
    )

    actual = HardClip(min_value=-100.0, max_value=100.0).transform(table)

    torch.testing.assert_close(
        actual.numerical,
        torch.tensor([[-100.0, -100.0], [100.0, 100.0]]),
    )
    assert actual.columns == table.columns
    assert actual.device == table.device
    assert repr(HardClip(min_value=-100.0, max_value=100.0)) == (
        "HardClip(min_value=-100.0, max_value=100.0)"
    )


def test_hard_clip_promotes_integer_input() -> None:
    table = TableTensor.from_tensor(torch.tensor([[-2, 2]]))

    actual = HardClip(min_value=-1.0, max_value=1.0).transform(table)

    assert actual.numerical.is_floating_point()
    torch.testing.assert_close(actual.numerical, torch.tensor([[-1.0, 1.0]]))


def test_hard_clip_rejects_invalid_configuration_and_stype() -> None:
    with pytest.raises(ValueError, match="min_value"):
        HardClip(min_value=1.0, max_value=-1.0)

    categorical = TableTensor(
        columns={"categorical": ("x",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0]], dtype=torch.int64),
            categories=(StringTensor.from_list(["a"]),),
        ),
    )
    with pytest.raises(ValueError, match="categorical"):
        HardClip(min_value=-1.0, max_value=1.0).transform(categorical)
