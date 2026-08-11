from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
import torch.nn.attention as torch_attention

from sdm.models import KumoRFM, TabICLv2
from sdm.models.kumorfm import model as kumorfm_module
from sdm.models.tabiclv2 import model as tabiclv2_module


class _StubInnerModel(torch.nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()


def _patch_flash_provider(
    monkeypatch: pytest.MonkeyPatch,
    active: str | None = None,
) -> dict[str, str | None]:
    state = {"active": active}
    monkeypatch.setattr(
        torch_attention,
        "activate_flash_attention_impl",
        lambda impl: state.update(active=impl),
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "list_flash_attention_impls",
        lambda: ["FA3"],
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "current_flash_attention_impl",
        lambda: state["active"],
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "restore_flash_attention_impl",
        lambda: state.update(active=None),
        raising=False,
    )
    return state


@pytest.fixture(
    params=[
        (TabICLv2, tabiclv2_module, "_TabICLv2"),
        (KumoRFM, kumorfm_module, "_KumoRFM"),
    ]
)
def model_cls(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Callable[..., torch.nn.Module]:
    model_cls, model_module, inner_model_name = request.param
    monkeypatch.setattr(model_module, inner_model_name, _StubInnerModel)
    return model_cls


def test_model_selects_flash_attention_impl(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
) -> None:
    state = _patch_flash_provider(monkeypatch, active="FA3")

    model_cls(pretrained=False, flash_attention_impl="FA2")
    assert state["active"] is None

    model_cls(pretrained=False, flash_attention_impl="FA3")
    assert state["active"] == "FA3"
    monkeypatch.setattr(
        torch_attention,
        "activate_flash_attention_impl",
        lambda impl: pytest.fail("provider was reactivated"),
    )

    model_cls(pretrained=False, flash_attention_impl="FA3")
    assert state["active"] == "FA3"

    state["active"] = "FA4"
    with pytest.raises(RuntimeError, match=r"FA4.*active"):
        model_cls(pretrained=False, flash_attention_impl="FA3")
    assert state["active"] == "FA4"

    state["active"] = None
    monkeypatch.setattr(
        torch_attention,
        "list_flash_attention_impls",
        list,
    )

    with pytest.raises(ValueError, match="not registered"):
        model_cls(pretrained=False, flash_attention_impl="FA3")
    assert state["active"] is None


def test_model_forces_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
) -> None:
    enabled: dict[str, bool] = {}
    _patch_flash_provider(monkeypatch)
    for backend in ("flash", "cudnn", "mem_efficient", "math"):
        monkeypatch.setattr(
            torch.backends.cuda,
            f"enable_{backend}_sdp",
            lambda value, backend=backend: enabled.update({backend: value}),
        )

    model_cls(
        pretrained=False,
        flash_attention_impl="FA3",
        force_flash_attention=True,
    )

    assert enabled == {
        "flash": True,
        "cudnn": False,
        "mem_efficient": False,
        "math": False,
    }


def test_model_rejects_unknown_flash_attention_impl(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
) -> None:
    with pytest.raises(ValueError, match="must be 'FA2' or 'FA3'"):
        model_cls(
            pretrained=False,
            flash_attention_impl=cast(Any, "FA4"),
        )


def test_fa3_requires_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
) -> None:
    monkeypatch.delattr(
        torch_attention,
        "activate_flash_attention_impl",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="provider registry"):
        model_cls(pretrained=False, flash_attention_impl="FA3")
