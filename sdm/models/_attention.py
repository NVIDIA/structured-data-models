import threading
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.attention as torch_attention

_CONFIGURATION_LOCK = threading.Lock()


def configure_flash_attention(
    impl: str | None,
    *,
    force: bool,
) -> None:
    if impl is None and not force:
        return

    with _CONFIGURATION_LOCK:
        if impl is not None:
            _activate_flash_attention_impl(impl)

        if force:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)


def _activate_flash_attention_impl(impl: str) -> None:
    activate = getattr(
        torch_attention,
        "activate_flash_attention_impl",
        None,
    )
    available = getattr(
        torch_attention,
        "list_flash_attention_impls",
        None,
    )
    current = getattr(
        torch_attention,
        "current_flash_attention_impl",
        None,
    )
    if not all(
        callable(function) for function in (activate, available, current)
    ):
        raise RuntimeError(
            "'flash_attention_impl' requires a PyTorch release with the "
            "Flash Attention provider registry"
        )
    activate = cast(Callable[[str], None], activate)
    available = cast(Callable[[], list[str]], available)
    current = cast(Callable[[], str | None], current)

    implementations = available()
    if impl not in implementations:
        raise ValueError(
            f"Flash Attention implementation {impl!r} is not "
            f"registered with PyTorch (available: {implementations})"
        )

    active = current()
    if active == impl:
        return
    if active is not None:
        raise RuntimeError(
            f"Cannot activate Flash Attention implementation {impl!r} "
            f"because {active!r} is already active. Provider activation "
            f"is process-wide."
        )

    activate(impl)
