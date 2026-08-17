import warnings

import torch

from sdm import CategoricalTensor

aten = torch.ops.aten


def test_is_pinned_exact_overload_accepts_omitted_and_explicit_device() -> (
    None
):
    tensor = CategoricalTensor(
        code=torch.tensor([[0]], dtype=torch.int32),
        categories=(torch.tensor([1]),),
    )

    assert not aten.is_pinned.default(tensor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert not aten.is_pinned.default(tensor, torch.device("cpu"))
