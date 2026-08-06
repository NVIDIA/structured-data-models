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
) -> TabICLv2:
    monkeypatch.setattr(
        tabiclv2_module,
        "_TabICLv2",
        _StubInnerModel,
    )
    return TabICLv2(
        pretrained=False,
        flash_attention_impl=impl,
    )


@pytest.mark.parametrize(
    ("model_cls", "model_module", "inner_model_name"),
    [
        (TabICLv2, tabiclv2_module, "_TabICLv2"),
        (KumoRFM, kumorfm_module, "_KumoRFM"),
    ],
)
@pytest.mark.parametrize(
    ("flash_attention_impl", "expected"),
    [(None, []), ("FA3", ["FA3"])],
)
def test_model_configures_flash_attention_impl(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: Callable[..., torch.nn.Module],
    model_module: object,
    inner_model_name: str,
    flash_attention_impl: str | None,
    expected: list[str],
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch)
    monkeypatch.setattr(model_module, inner_model_name, _StubInnerModel)

    model = model_cls(
        pretrained=False,
        flash_attention_impl=flash_attention_impl,
    )

    assert not model.training
    assert registry.activations == expected


def test_model_does_not_activate_after_failed_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _patch_flash_provider_registry(monkeypatch)

    class _FailingInnerModel(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("initialization failed")

    monkeypatch.setattr(
        tabiclv2_module,
        "_TabICLv2",
        _FailingInnerModel,
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        TabICLv2(pretrained=False, flash_attention_impl="FA3")

    assert registry.activations == []


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
