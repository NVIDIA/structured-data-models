from collections.abc import Callable

import torch
from torch import Tensor


def google_state_dict(
    module: torch.nn.Module,
    value_fn: Callable[[str, Tensor], Tensor],
) -> dict[str, Tensor]:
    """Create generic-module state using the corresponding TabFM names."""
    state = module.state_dict()
    keys = set(state)

    def block_name(name: str) -> str:
        name = name.replace(".transformer_1.", ".mab1.")
        name = name.replace(".transformer_2.", ".mab2.")
        if name.startswith("tf_col."):
            name = name.replace("tf_col.", "tf_col.blocks.", 1)
        return name

    for key, tensor in state.items():
        if key.endswith("attn.qkv_lin.weight"):
            prefix = block_name(key.removesuffix("qkv_lin.weight"))
            width = tensor.size(0) // 3
            state[key] = torch.cat(
                [
                    value_fn(
                        prefix + f"{projection}_proj.weight",
                        tensor[:width],
                    )
                    for projection in "qkv"
                ]
            ).to(tensor.dtype)
            continue
        if key.endswith("attn.qkv_lin.bias"):
            prefix = block_name(key.removesuffix("qkv_lin.bias"))
            width = tensor.size(0) // 3
            state[key] = torch.cat(
                [
                    value_fn(
                        prefix + f"{projection}_proj.bias",
                        tensor[:width],
                    )
                    for projection in "qkv"
                ]
            ).to(tensor.dtype)
            continue

        google_key = key
        google_key = google_key.replace(
            ".shared_attention_norm.", ".pre_attn_ln."
        )
        google_key = google_key.replace(
            ".post_attention_norm.", ".post_attn_ln."
        )
        google_key = block_name(google_key)
        google_key = google_key.replace(".inducing_points", ".ind_vectors")
        google_key = google_key.replace(".attn.out_lin.", ".attn.out_proj.")
        google_key = google_key.replace(".mlp.module.0.", ".pre_ff_ln.")
        google_key = google_key.replace(
            ".mlp.module.1.value_lin.", ".linear1."
        )
        google_key = google_key.replace(
            ".mlp.module.1.gate_lin.", ".linear1_gate."
        )
        google_key = google_key.replace(".mlp.module.1.out_lin.", ".linear2.")
        google_key = google_key.replace(".mlp.module.2.", ".post_ff_ln.")
        if ".attn.query_transform." in key and key.endswith(".weight"):
            prefix, index = key.rsplit(".", maxsplit=2)[:2]
            has_rope = f"{prefix}.0.inv_freq" in keys
            google_key = key.split("attn.query_transform.", 1)[0] + (
                "attn.query_ln.weight"
                if int(index) == int(has_rope)
                else "attn.per_dim_scale"
            )
        elif ".attn.key_transform." in key and key.endswith(".weight"):
            google_key = key.split("attn.key_transform.", 1)[0] + (
                "attn.key_ln.weight"
            )
        elif key.endswith(".inv_freq"):
            google_key = "rope.freqs"
        state[key] = value_fn(google_key, tensor).to(tensor.dtype)
    return state


def frozen_google_state_dict(
    module: torch.nn.Module,
    *,
    rope_value: float = 0.75,
) -> dict[str, Tensor]:
    """Create deterministic state using pinned Google TabFM names."""

    def value(name: str, tensor: Tensor) -> Tensor:
        if name.endswith("rope.freqs"):
            values = torch.full(
                (tensor.numel(),),
                rope_value,
                device=tensor.device,
                dtype=torch.float32,
            )
        elif name.endswith("_ln.weight") or name == "ln_w.weight":
            offset = (sum(name.encode()) % 7) * 0.05
            values = torch.linspace(
                0.8,
                1.2,
                tensor.numel(),
                device=tensor.device,
            ).add_(offset)
        elif name.endswith("per_dim_scale"):
            values = torch.linspace(
                -0.3,
                0.2,
                tensor.numel(),
                device=tensor.device,
            )
        elif name.endswith("ind_vectors"):
            offset = (sum(name.encode()) % 5 - 2) * 0.02
            values = torch.linspace(
                -0.25,
                0.25,
                tensor.numel(),
                device=tensor.device,
            ).add_(offset)
        else:
            offset = (sum(name.encode()) % 11 - 5) * 0.003
            start, end = (
                (-0.04, 0.04) if name.endswith(".bias") else (-0.15, 0.15)
            )
            values = torch.linspace(
                start,
                end,
                tensor.numel(),
                device=tensor.device,
            ).add_(offset)
        return values.reshape(tensor.shape)

    return google_state_dict(module, value)
