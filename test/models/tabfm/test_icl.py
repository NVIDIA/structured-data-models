import pytest
import torch
from test.models.tabfm._testing import google_state_dict
from torch import Tensor

from sdm.models.tabfm.icl import _ICLearning
from sdm.testing import withCUDA

_GOLDEN_OUTPUTS = {
    (True, torch.float32): (
        "-.1554137915 .0185760036 .1925657988 -.1557042450 .0240286365 "
        ".2037615180 -.1557671428 .0243039504 .2043750286 -.1550834775 "
        ".0218387805 .1987610459 -.1560166627 .0231710654 .2023587972 "
        "-.1559757143 .0241944939 .2043646872 -.1562774479 .0273138657 "
        ".2109051794 -.1561222076 .0237614829 .2036451697"
    ),
    (True, torch.bfloat16): (
        "-.1552734375 .0186767578 .1923828125 -.15625 .0240478516 "
        ".2041015625 -.15625 .0241699219 .2041015625 -.1552734375 "
        ".0218505859 .19921875 -.15625 .0230712891 .2021484375 -.15625 "
        ".0242919922 .205078125 -.15625 .0272216797 .2109375 -.15625 "
        ".0238037109 .2041015625"
    ),
    (False, torch.float32): (
        ".0556138605 .0706657842 .0714303479 .0642918646 .0693681315 "
        ".0704170018 .0811308026 .0692408830"
    ),
    (False, torch.bfloat16): (
        ".0554199219 .0703125 .0712890625 .0639648438 .0693359375 "
        ".0703125 .0810546875 .0693359375"
    ),
}

_GOLDEN_REPRESENTATIONS = (
    "-.5 .25 .75 -1 1 -.75 .5 0 -.25 1.25 -.5 .5 .75 0 -1.25 1 "
    ".5 -1 .25 .75 -.75 .5 1 -.25 1.25 -.5 0 .5 -1 .75 -.25 1.25"
)


def _google_value(key: str, tensor: Tensor) -> Tensor:
    size = tensor.numel()
    values = torch.arange(
        size,
        device=tensor.device,
        dtype=torch.float32,
    ).reshape(tensor.shape)
    values = (values - (size - 1) / 2) * (0.3 / max(size - 1, 1))
    values = values + (sum(map(ord, key)) % 7 - 3) * 0.01
    if key.endswith("ln.weight"):
        values = values + 1
    return values


def _google_state(module: torch.nn.Module) -> dict[str, Tensor]:
    return google_state_dict(module, _google_value)


def _module(
    is_classifier: bool,
    *,
    max_classes: int = 3,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> _ICLearning:
    return _ICLearning(
        channels=4,
        num_blocks=1,
        num_heads=2,
        feedforward_channels=6,
        decoder_hidden_channels=5,
        is_classifier=is_classifier,
        max_classes=max_classes,
        device=device,
        dtype=dtype,
    )


@withCUDA
@pytest.mark.parametrize("is_classifier", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_icl_matches_google_golden(
    device: torch.device,
    is_classifier: bool,
    dtype: torch.dtype,
) -> None:
    module = _module(is_classifier, device=device, dtype=dtype)
    module.load_state_dict(_google_state(module), strict=True)
    representations = torch.tensor(
        [float(value) for value in _GOLDEN_REPRESENTATIONS.split()],
        device=device,
        dtype=dtype,
    ).view(2, 4, 4)
    targets = torch.tensor(
        [[-100, 99, -7, 4], [0, 2, 1, -100]]
        if is_classifier
        else [[0.5, 99, -7, 4], [1.25, -0.75, 0.25, -100]],
        device=device,
        dtype=torch.float32 if is_classifier else torch.float64,
    )

    output = module(
        representations,
        targets,
        torch.tensor([1, 3], device=device),
    )

    # Frozen from google-research/tabfm@b8a8b090's PyTorch ICLearning.
    expected = output.new_tensor(
        [
            float(value)
            for value in _GOLDEN_OUTPUTS[is_classifier, dtype].split()
        ]
    ).view_as(output)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("is_classifier", [True, False])
def test_icl_isolates_queries_and_uses_context(
    is_classifier: bool,
) -> None:
    module = _module(is_classifier, max_classes=10)
    module.load_state_dict(_google_state(module))
    representations = torch.arange(48, dtype=torch.float32).view(3, 4, 4) / 20
    context_size = torch.tensor([0, 2, 4])
    targets = (
        torch.tensor([[0, 1, 2, 0], [0, 2, -100, 99], [2, 1, 0, 2]])
        if is_classifier
        else torch.tensor(
            [
                [0.0, 1.0, 2.0, 0.0],
                [0.5, -0.5, -100.0, 99.0],
                [2.0, 1.0, 0.0, -1.0],
            ]
        )
    )
    rows = torch.arange(4)[None, :]
    query = rows >= context_size[:, None]
    sentinel_values = (
        [-100, 99, -7, 4]
        if is_classifier
        else [float("nan"), float("inf"), -float("inf"), float("nan")]
    )
    sentinels = torch.tensor(
        [sentinel_values],
        dtype=targets.dtype,
    ).expand_as(targets)

    expected = module(representations, targets, context_size)
    changed_targets = torch.where(query, sentinels, targets)
    torch.testing.assert_close(
        module(representations, changed_targets, context_size),
        expected,
        rtol=0,
        atol=0,
    )

    changed_representations = representations.clone()
    changed_representations[1, 2] += torch.tensor([2.0, -1.0, 0.5, 1.0])
    changed = module(changed_representations, targets, context_size)
    torch.testing.assert_close(
        changed[1, [0, 1, 3]],
        expected[1, [0, 1, 3]],
        rtol=0,
        atol=0,
    )
    assert not torch.allclose(changed[1, 2], expected[1, 2])

    changed_targets = targets.clone()
    changed_targets[1, 0] = 1 if is_classifier else 2.5
    changed = module(representations, changed_targets, context_size)
    assert not torch.allclose(changed[1, 2:], expected[1, 2:])
    assert expected.shape == (3, 4, 10 if is_classifier else 1)


def test_icl_rejects_empty_classification_head() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        _module(True, max_classes=0)
