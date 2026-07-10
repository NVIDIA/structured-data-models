from typing import Any, cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    CategoryShuffle,
    MeanImpute,
    StandardScale,
    TargetDispatch,
)


def _classification_target() -> TableTensor:
    return TableTensor(
        columns={"categorical": ("label",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [2]], dtype=torch.int64),
            categories=(StringTensor.from_list(["a", "b", "c"]),),
        ),
    )


def test_target_dispatch_inverts_complete_class_score_head() -> None:
    torch.manual_seed(1)
    dispatch = TargetDispatch(
        classification=CategoryShuffle(method="random"),
        regression=StandardScale(),
    )

    transformed = dispatch.fit_transform(_classification_target())

    assert dispatch.get_extra_state() == "classification"
    assert transformed.categorical.size(-1) == 1
    selected = cast(CategoryShuffle, dispatch.selected)
    canonical = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    raw = torch.full((2, 10), -100.0, dtype=canonical.dtype)
    raw[..., selected.permutations] = canonical

    restored = dispatch.inverse_transform(TableTensor.from_tensor(raw))

    assert restored.size() == canonical.size()
    torch.testing.assert_close(
        restored.numerical,
        canonical,
        rtol=0,
        atol=0,
    )


def test_target_dispatch_regression_roundtrip_and_refit() -> None:
    dispatch = TargetDispatch(
        classification=CategoryShuffle(),
        regression=StandardScale(),
    )
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))

    transformed = dispatch.fit_transform(target)
    restored = dispatch.inverse_transform(transformed)

    assert dispatch.get_extra_state() == "regression"
    torch.testing.assert_close(restored.numerical, target.numerical)

    dispatch.fit(_classification_target())
    assert dispatch.get_extra_state() == "classification"


def test_target_dispatch_rejects_invalid_routes_and_targets() -> None:
    with pytest.raises(TypeError, match=r"invertible.*classification"):
        TargetDispatch(classification=MeanImpute())
    with pytest.raises(ValueError, match="at least one route"):
        TargetDispatch()

    dispatch = TargetDispatch(regression=StandardScale())
    with pytest.raises(ValueError, match="exactly one column"):
        dispatch.fit(TableTensor.from_tensor(torch.ones(2, 2)))
    with pytest.raises(ValueError, match="no 'classification' route"):
        dispatch.fit(_classification_target())

    with pytest.raises(ValueError, match="unconfigured 'classification'"):
        dispatch.set_extra_state(cast(Any, "classification"))
