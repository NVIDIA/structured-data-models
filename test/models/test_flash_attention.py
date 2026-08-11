from collections.abc import Callable

import pytest
import torch
import torch.nn.attention as torch_attention

from sdm.models import KumoRFM, TabICLv2
from sdm.models.kumorfm import model as kumorfm_module
from sdm.models.tabiclv2 import model as tabiclv2_module


class _StubInnerModel(torch.nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()


class _FlashProviderRegistry:
    def __init__(self, active: str | None = None) -> None:
        self.active = active
        self.activations: list[str] = []

    def activate(self, impl: str) -> None:
        self.activations.append(impl)
        self.active = impl

    @staticmethod
    def available() -> list[str]:
        return ["FA3", "FA4"]

    def current(self) -> str | None:
        return self.active


class _SDPBackendRouting:
    def __init__(self) -> None:
        self.flash = True
        self.cudnn = True
        self.mem_efficient = True
        self.math = True

    @property
    def state(self) -> tuple[bool, bool, bool, bool]:
        return self.flash, self.cudnn, self.mem_efficient, self.math


def _patch_sdp_backend_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> _SDPBackendRouting:
    routing = _SDPBackendRouting()
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_flash_sdp",
        lambda enabled: setattr(routing, "flash", enabled),
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_cudnn_sdp",
        lambda enabled: setattr(routing, "cudnn", enabled),
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_mem_efficient_sdp",
        lambda enabled: setattr(routing, "mem_efficient", enabled),
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_math_sdp",
        lambda enabled: setattr(routing, "math", enabled),
    )
    return routing


def _patch_flash_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active: str | None = None,
) -> _FlashProviderRegistry:
    registry = _FlashProviderRegistry(active)
    monkeypatch.setattr(
        torch_attention,
        "activate_flash_attention_impl",
        registry.activate,
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "list_flash_attention_impls",
        registry.available,
        raising=False,
    )
    monkeypatch.setattr(
        torch_attention,
        "current_flash_attention_impl",
        registry.current,
        raising=False,
    )
    return registry


def _make_tabiclv2(
    monkeypatch: pytest.MonkeyPatch,
    impl: str,
    *,
    force: bool = False,
) -> TabICLv2:
    monkeypatch.setattr(
        tabiclv2_module,
        "_TabICLv2",
        _StubInnerModel,
    )
    return TabICLv2(
        pretrained=False,
        flash_attention_impl=impl,
        force_flash_attention=force,
    )


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
        "expected_activations",
        "expected_routing",
    ),
    [
        (None, False, [], (True, True, True, True)),
        ("FA3", False, ["FA3"], (True, True, True, True)),
        (None, True, [], (True, False, False, False)),
        ("FA3", True, ["FA3"], (True, False, False, False)),
    ],
)
def test_model_configures_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
    model_module: object,
    inner_model_name: str,
    flash_attention_impl: str | None,
    force_flash_attention: bool,
    expected_activations: list[str],
    expected_routing: tuple[bool, bool, bool, bool],
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch)
    routing = _patch_sdp_backend_routing(monkeypatch)
    monkeypatch.setattr(model_module, inner_model_name, _StubInnerModel)

    model = model_cls(
        pretrained=False,
        flash_attention_impl=flash_attention_impl,
        force_flash_attention=force_flash_attention,
    )

    assert not model.training
    assert registry.activations == expected_activations
    assert routing.state == expected_routing


def test_model_does_not_activate_after_failed_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch)
    routing = _patch_sdp_backend_routing(monkeypatch)

    class _FailingInnerModel(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(
        tabiclv2_module,
        "_TabICLv2",
        _FailingInnerModel,
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        TabICLv2(
            pretrained=False,
            flash_attention_impl="FA3",
            force_flash_attention=True,
        )

    assert registry.activations == []
    assert routing.state == (True, True, True, True)


def test_activate_flash_attention_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch)

    _make_tabiclv2(monkeypatch, "FA3")

    assert registry.activations == ["FA3"]


def test_activate_flash_attention_impl_without_tracking_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch)
    monkeypatch.setattr(
        torch_attention,
        "activate_flash_attention_impl",
        registry.activations.append,
    )

    _make_tabiclv2(monkeypatch, "FA3")

    assert registry.active is None
    assert registry.activations == ["FA3"]


def test_activate_flash_attention_impl_reuses_active_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch, active="FA3")

    _make_tabiclv2(monkeypatch, "FA3")

    assert registry.activations == []


def test_activate_flash_attention_impl_rejects_provider_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch, active="FA4")

    with pytest.raises(RuntimeError, match=r"FA4.*already active"):
        _make_tabiclv2(monkeypatch, "FA3")

    assert registry.activations == []


def test_activate_flash_attention_impl_validates_before_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch, active="FA3")

    with pytest.raises(ValueError, match=r"UNKNOWN.*not registered"):
        _make_tabiclv2(monkeypatch, "UNKNOWN")

    assert registry.activations == []


def test_activate_flash_attention_impl_requires_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(
        torch_attention,
        "activate_flash_attention_impl",
        raising=False,
    )
    monkeypatch.delattr(
        torch_attention,
        "list_flash_attention_impls",
        raising=False,
    )
    monkeypatch.delattr(
        torch_attention,
        "current_flash_attention_impl",
        raising=False,
    )

    with pytest.raises(RuntimeError, match="provider registry"):
        _make_tabiclv2(monkeypatch, "FA3")


def test_force_flash_attention_does_not_require_provider_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(
        torch_attention,
        "activate_flash_attention_impl",
        raising=False,
    )
    monkeypatch.delattr(
        torch_attention,
        "list_flash_attention_impls",
        raising=False,
    )
    monkeypatch.delattr(
        torch_attention,
        "current_flash_attention_impl",
        raising=False,
    )
    routing = _patch_sdp_backend_routing(monkeypatch)
    monkeypatch.setattr(tabiclv2_module, "_TabICLv2", _StubInnerModel)

    TabICLv2(pretrained=False, force_flash_attention=True)

    assert routing.state == (True, False, False, False)
