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


class _FlashProvider:
    def __init__(self, active: str | None = None) -> None:
        self.active = active

    def activate(self, impl: str) -> None:
        self.active = impl

    def restore(self) -> None:
        self.active = None


class _SDPBackendRouting:
    def __init__(self) -> None:
        self.flash = True
        self.cudnn = True
        self.mem_efficient = True
        self.math = True

    @property
    def state(self) -> tuple[bool, bool, bool, bool]:
        return self.flash, self.cudnn, self.mem_efficient, self.math


def _patch_flash_provider(
    monkeypatch: pytest.MonkeyPatch,
    active: str | None,
) -> _FlashProvider:
    provider = _FlashProvider(active)
    monkeypatch.setattr(
        torch_attention,
        "activate_flash_attention_impl",
        provider.activate,
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "current_flash_attention_impl",
        lambda: provider.active,
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "restore_flash_attention_impl",
        provider.restore,
        raising=False,
    )
    return provider


def _patch_sdp_backend_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> _SDPBackendRouting:
    routing = _SDPBackendRouting()
    for name, attribute in [
        ("enable_flash_sdp", "flash"),
        ("enable_cudnn_sdp", "cudnn"),
        ("enable_mem_efficient_sdp", "mem_efficient"),
        ("enable_math_sdp", "math"),
    ]:
        monkeypatch.setattr(
            torch.backends.cuda,
            name,
            lambda enabled, attribute=attribute: setattr(
                routing,
                attribute,
                enabled,
            ),
        )
    return routing


@pytest.mark.parametrize(
    ("model_cls", "model_module", "inner_model_name"),
    [
        (TabICLv2, tabiclv2_module, "_TabICLv2"),
        (KumoRFM, kumorfm_module, "_KumoRFM"),
    ],
)
@pytest.mark.parametrize(
    (
        "flash_attention_impl",
        "force_flash_attention",
        "initial_provider",
        "expected_provider",
        "expected_routing",
    ),
    [
        (None, False, None, None, (True, True, True, True)),
        ("FA2", False, "FA3", None, (True, True, True, True)),
        ("FA3", False, None, "FA3", (True, True, True, True)),
        ("FA2", True, "FA3", None, (True, False, False, False)),
        ("FA3", True, None, "FA3", (True, False, False, False)),
    ],
)
def test_model_configures_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
    model_module: object,
    inner_model_name: str,
    flash_attention_impl: str | None,
    force_flash_attention: bool,
    initial_provider: str | None,
    expected_provider: str | None,
    expected_routing: tuple[bool, bool, bool, bool],
) -> None:
    provider = _patch_flash_provider(monkeypatch, initial_provider)
    routing = _patch_sdp_backend_routing(monkeypatch)
    monkeypatch.setattr(model_module, inner_model_name, _StubInnerModel)

    model = model_cls(
        pretrained=False,
        flash_attention_impl=flash_attention_impl,
        force_flash_attention=force_flash_attention,
    )

    assert not model.training
    assert provider.active == expected_provider
    assert routing.state == expected_routing


def test_model_rejects_unknown_flash_attention_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tabiclv2_module, "_TabICLv2", _StubInnerModel)
    with pytest.raises(ValueError, match="must be 'FA2' or 'FA3'"):
        TabICLv2(
            pretrained=False,
            flash_attention_impl=cast(Any, "FA4"),
        )


def test_fa3_requires_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tabiclv2_module, "_TabICLv2", _StubInnerModel)
    monkeypatch.delattr(
        torch_attention,
        "activate_flash_attention_impl",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="provider registry"):
        TabICLv2(pretrained=False, flash_attention_impl="FA3")


def test_model_does_not_configure_after_failed_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _patch_flash_provider(monkeypatch, None)
    routing = _patch_sdp_backend_routing(monkeypatch)

    class _FailingInnerModel(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(tabiclv2_module, "_TabICLv2", _FailingInnerModel)

    with pytest.raises(RuntimeError, match="initialization failed"):
        TabICLv2(
            pretrained=False,
            flash_attention_impl="FA3",
            force_flash_attention=True,
        )

    assert provider.active is None
    assert routing.state == (True, True, True, True)
