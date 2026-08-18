"""Load TabFM v1.0.0-format SafeTensor checkpoints.

Released weights use ``tabfm-non-commercial-v1.0`` and may not be used for
commercial or production purposes or redistributed. See the pinned
`license <https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/LICENSE>`_.
No weights are bundled with this package. Local files are not authenticated
as released artifacts.
"""

import re
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file
from torch import Tensor

from sdm.models.tabfm.model import _TabFM

_Task = Literal["classification", "regression"]


def _block_source(
    suffix: str,
    *,
    prefix: str,
    rope_key: str | None,
) -> tuple[Literal["direct", "qkv"], tuple[str, ...]]:
    """Map one SDM transformer-block suffix to released parameter names."""
    if suffix.startswith("mlp.0."):
        suffix = suffix.replace("mlp.0.", "pre_ff_ln.", 1)
    elif suffix.startswith("mlp.1.up_lin."):
        suffix = suffix.replace("mlp.1.up_lin.", "linear1.", 1)
    elif suffix.startswith("mlp.1.gate_lin."):
        suffix = suffix.replace("mlp.1.gate_lin.", "linear1_gate.", 1)
    elif suffix.startswith("mlp.1.down_lin."):
        suffix = suffix.replace("mlp.1.down_lin.", "linear2.", 1)
    elif suffix.startswith("mlp.2."):
        suffix = suffix.replace("mlp.2.", "post_ff_ln.", 1)
    elif suffix.startswith(("query_norm.", "key_value_norm.")):
        suffix = re.sub(
            r"^(query_norm|key_value_norm)\.", "pre_attn_ln.", suffix
        )
    elif suffix.startswith("post_attn_norm."):
        suffix = suffix.replace("post_attn_norm.", "post_attn_ln.", 1)
    elif suffix.startswith("attn.qkv_lin."):
        parameter = suffix.rsplit(".", 1)[-1]
        return "qkv", tuple(
            f"{prefix}.attn.{projection}_proj.{parameter}"
            for projection in ("q", "k", "v")
        )
    elif suffix.startswith("attn.out_lin."):
        suffix = suffix.replace("attn.out_lin.", "attn.out_proj.", 1)
    elif suffix.startswith("attn.query_transform."):
        index, parameter = suffix.removeprefix("attn.query_transform.").split(
            ".", 1
        )
        if parameter == "inv_freq":
            if rope_key is None or index != "0":
                raise ValueError(f"Unexpected query transform {suffix!r}")
            return "direct", (rope_key,)
        norm_index = int(rope_key is not None)
        if parameter != "weight":
            raise ValueError(f"Unexpected query transform {suffix!r}")
        if index == str(norm_index):
            source = "query_ln.weight"
        elif index == str(norm_index + 1):
            source = "per_dim_scale"
        else:
            raise ValueError(f"Unexpected query transform {suffix!r}")
        return "direct", (f"{prefix}.attn.{source}",)
    elif suffix.startswith("attn.key_transform."):
        index, parameter = suffix.removeprefix("attn.key_transform.").split(
            ".", 1
        )
        if parameter == "inv_freq":
            if rope_key is None or index != "0":
                raise ValueError(f"Unexpected key transform {suffix!r}")
            return "direct", (rope_key,)
        if parameter != "weight" or index != str(int(rope_key is not None)):
            raise ValueError(f"Unexpected key transform {suffix!r}")
        return "direct", (f"{prefix}.attn.key_ln.weight",)

    return "direct", (f"{prefix}.{suffix}",)


def _source(
    target_key: str,
) -> tuple[Literal["direct", "qkv", "class_embedding"], tuple[str, ...]]:
    """Return the released checkpoint source for one SDM state key."""
    cell_names = {
        "num_freq": "fourier_frequencies",
        "cat_freq": "fourier_frequencies_cat",
        "num_lin": "in_linear",
        "cat_lin": "in_linear_cat",
    }
    if target_key.startswith("cell_embedding."):
        name, separator, suffix = target_key.removeprefix(
            "cell_embedding."
        ).partition(".")
        source = cell_names[name]
        return "direct", (f"cell_embedder.{source}{separator}{suffix}",)

    if target_key == "row_embedding.readout_token":
        return "direct", ("cls_tokens",)
    if target_key.startswith("row_embedding.y_emb."):
        suffix = target_key.removeprefix("row_embedding.y_emb.")
        return "direct", (f"cell_embedder.y_embedder_lookup.{suffix}",)
    if target_key.startswith("row_embedding.y_mlp."):
        suffix = target_key.removeprefix("row_embedding.y_mlp.")
        index, parameter = suffix.split(".", 1)
        return "direct", (
            "cell_embedder.y_embedder_lookup.layers."
            f"{int(index) // 2}.{parameter}",
        )

    match = re.match(
        r"row_embedding\.col_blocks\.(\d+)\.(\d+)\.(.+)", target_key
    )
    if match is not None:
        repeat, layer, suffix = match.groups()
        root = "col_embedder" if repeat == "0" else "col_embedder_2"
        suffix = suffix.replace("inducing_block.", "mab1.", 1)
        suffix = suffix.replace("output_block.", "mab2.", 1)
        suffix = suffix.replace("inducing_points", "ind_vectors", 1)
        block, separator, remainder = suffix.partition(".")
        prefix = f"{root}.tf_col.blocks.{layer}.{block}"
        if block == "ind_vectors":
            return "direct", (prefix,)
        return _block_source(remainder, prefix=prefix, rope_key=None)

    match = re.match(
        r"row_embedding\.row_blocks\.(\d+)\.(\d+)\.(.+)", target_key
    )
    if match is not None:
        repeat, layer, suffix = match.groups()
        root = "row_interactor" if repeat == "0" else "row_interactor_2"
        prefix = f"{root}.tf_row.blocks.{layer}"
        # SDM shares the non-learned RoPE between the two repeated stages.
        rope_key = "row_interactor.tf_row.rope.freqs"
        return _block_source(suffix, prefix=prefix, rope_key=rope_key)

    match = re.match(
        r"row_embedding\.col_projections\.(\d+)\.(\d+)\.(.+)", target_key
    )
    if match is not None:
        repeat, index, suffix = match.groups()
        root = "col_embedder" if repeat == "0" else "col_embedder_2"
        module = "out_w" if index == "0" else "ln_w"
        return "direct", (f"{root}.{module}.{suffix}",)

    match = re.match(r"row_embedding\.row_norms\.(\d+)\.(.+)", target_key)
    if match is not None:
        repeat, suffix = match.groups()
        root = "row_interactor" if repeat == "0" else "row_interactor_2"
        return "direct", (f"{root}.out_ln.{suffix}",)

    if target_key.startswith("icl_block.y_emb."):
        return "class_embedding", (
            "icl_predictor.y_encoder.projection.weight",
            "icl_predictor.y_encoder.projection.bias",
        )
    if target_key.startswith("icl_block.y_mlp."):
        suffix = target_key.removeprefix("icl_block.y_mlp.")
        index, parameter = suffix.split(".", 1)
        return "direct", (
            f"icl_predictor.y_encoder.layers.{int(index) // 2}.{parameter}",
        )

    match = re.match(r"icl_block\.layers\.(\d+)\.(.+)", target_key)
    if match is not None:
        layer, suffix = match.groups()
        return _block_source(
            suffix,
            prefix=f"icl_predictor.tf_icl.blocks.{layer}",
            rope_key=None,
        )
    if target_key.startswith("icl_block.norm."):
        return "direct", (
            target_key.replace("icl_block.norm.", "icl_predictor.ln.", 1),
        )
    if target_key.startswith("icl_block.head."):
        suffix = target_key.removeprefix("icl_block.head.")
        index, parameter = suffix.split(".", 1)
        return "direct", (
            f"icl_predictor.decoder.layers.{int(index) // 2}.{parameter}",
        )

    raise KeyError(f"Unmapped TabFM parameter {target_key!r}")


def _remap_state_dict(
    source: dict[str, Tensor],
    expected: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Translate released TabFM tensors into the SDM state layout."""
    first_rope = "row_interactor.tf_row.rope.freqs"
    second_rope = "row_interactor_2.tf_row.rope.freqs"
    if first_rope in source or second_rope in source:
        missing_rope = [
            key for key in (first_rope, second_rope) if key not in source
        ]
        if missing_rope:
            raise KeyError(f"Checkpoint is missing {missing_rope!r}")
        if not torch.equal(source[first_rope], source[second_rope]):
            raise ValueError("TabFM checkpoint row RoPE tensors do not match")
        source.pop(second_rope)

    remapped: dict[str, Tensor] = {}
    converted: dict[str, Tensor] = {}
    for target_key in expected:
        kind, source_keys = _source(target_key)
        if kind == "direct" and source_keys[0] in converted:
            remapped[target_key] = converted[source_keys[0]]
            continue

        missing = [key for key in source_keys if key not in source]
        if missing:
            raise KeyError(
                f"Checkpoint is missing {missing!r} required for "
                f"{target_key!r}"
            )
        tensors = tuple(source.pop(key) for key in source_keys)
        if kind == "direct":
            value = tensors[0]
        elif kind == "qkv":
            # Pack the released Q/K/V projection outputs along one axis.
            value = torch.cat(tensors, dim=0)
        else:
            assert kind == "class_embedding"
            weight, bias = tensors
            # Convert one-hot linear weights into embedding lookup rows.
            value = weight.t() + bias
        remapped[target_key] = value
        if kind == "direct":
            converted[source_keys[0]] = value

    if source:
        raise ValueError(f"Unmapped checkpoint parameters: {sorted(source)!r}")
    return remapped


def _load_tabfm_v1_0_0(
    checkpoint_path: str | Path,
    *,
    task: _Task,
    device: torch.device | str | None = None,
) -> _TabFM:
    """Load a TabFM v1.0.0-format checkpoint."""
    target_device = torch.device("cpu" if device is None else device)
    model = _TabFM(
        num_classes=10 if task == "classification" else 0,
        device=target_device,
    )
    source = load_file(checkpoint_path, device=str(target_device))
    model.load_state_dict(
        _remap_state_dict(source, model.state_dict()),
        strict=True,
    )
    return model
