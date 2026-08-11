from typing import Literal

import torch
import torch.nn.attention as torch_attention


def configure_flash_attention(
    impl: Literal["FA2", "FA3"] | None,
    *,
    force: bool,
) -> None:
    if impl == "FA2":
        current = getattr(
            torch_attention,
            "current_flash_attention_impl",
            None,
        )
        restore = getattr(
            torch_attention,
            "restore_flash_attention_impl",
            None,
        )
        if callable(current) and current() is not None:
            if not callable(restore):
                raise RuntimeError("PyTorch cannot restore Flash Attention 2")
            restore()
    elif impl == "FA3":
        activate = getattr(
            torch_attention,
            "activate_flash_attention_impl",
            None,
        )
        if not callable(activate):
            raise RuntimeError(
                "Flash Attention 3 requires PyTorch's provider registry"
            )
        activate("FA3")
    elif impl is not None:
        raise ValueError("'flash_attention_impl' must be 'FA2' or 'FA3'")

    if force:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
