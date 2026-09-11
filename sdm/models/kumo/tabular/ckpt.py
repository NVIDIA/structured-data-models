import re

import torch
from torch import Tensor


def remap_ckpt(  # noqa: D103
    ckpt: dict[str, Tensor],
    is_classifier: bool,
    num_layers: int,
) -> dict[str, Tensor]:

    out: dict[str, Tensor] = {}
    for key, value in ckpt.items():
        if key == "prediction_head.alphas":
            expected = torch.arange(1, 1000, device=value.device) / 1000
            if not torch.allclose(value, expected.to(value.dtype)):
                raise ValueError(
                    "Expected regression quantiles 0.001 to 0.999"
                )
            continue

        if key.startswith(("y_encoder.", "icl_y_encoder.")):
            prefix = (
                "row_embedding"
                if key.startswith("y_encoder.")
                else "icl_block"
            )
            if is_classifier and key.endswith("bias"):
                continue
            if is_classifier:
                out[f"{prefix}.y_emb.weight"] = (
                    value.T + ckpt[key.removesuffix("weight") + "bias"]
                )
            else:
                out[f"{prefix}.y_lin.{key.rsplit('.', 1)[-1]}"] = value
            continue

        if key == "encoder.rope.inv_freq":
            for layer in range(num_layers):
                for transform in ("query_transform", "key_transform"):
                    out[
                        f"row_embedding.row_blocks.{layer}.attn."
                        f"{transform}.0.inv_freq"
                    ] = value
            continue

        if key.startswith("cell_embedding."):
            key = f"row_embedding.{key}"
            for old, new in (
                ("frequencies_num", "num_freq"),
                ("frequencies_cat", "cat_freq"),
                ("lin_num", "num_lin"),
                ("lin_cat", "cat_lin"),
                ("lin_missing", "nan_lin"),
            ):
                key = key.replace(old, new)
        elif key == "encoder.cls_tokens":
            key = "row_embedding.readout_token"
        elif key.startswith("encoder.stages."):
            match = re.fullmatch(
                r"encoder\.stages\.(\d+)\.blocks\.0\.(.*)", key
            )
            assert match is not None, key
            stage = int(match[1])
            kind = "col_blocks" if stage % 2 == 0 else "row_blocks"
            key = f"row_embedding.{kind}.{stage // 2}.{match[2]}"
            key = key.replace("inducing_vectors", "inducing_points")
            key = key.replace("to_inducing", "inducing_block")
            key = key.replace("from_inducing", "output_block")
        elif key.startswith("encoder.norm."):
            key = key.replace("encoder.norm.", "row_embedding.norm.")
        elif key.startswith("cls_downproject."):
            key = key.replace("cls_downproject.", "row_project.")
        elif key.startswith("icl.blocks."):
            key = key.replace("icl.blocks.", "icl_block.layers.")
        elif key.startswith("icl.norm."):
            key = key.replace("icl.norm.", "icl_block.norm.")
        elif key.startswith("prediction_head.mlp."):
            key = key.replace("prediction_head.mlp.", "icl_block.head.")

        for old, new in (
            (".q_norm.", ".query_norm."),
            (".kv_norm.", ".key_value_norm."),
            (".mlp_norm.", ".mlp.0."),
            (".lin1.", ".mlp.1."),
            (".lin2.", ".mlp.3."),
            (".gated_per_head_logn_scale.", ".query_scaling."),
            (".per_head_logn_scale.", ".query_scaling."),
        ):
            key = key.replace(old, new)

        out[key] = value

    return out
