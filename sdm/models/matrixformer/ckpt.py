# Copyright 2026 The MatrixFormer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

_ENCODER = {
    "feature_encoder.observed_linear_layer.": "value_mlp.",
    "feature_encoder.missingness_embedding.": "missing_embedding.",
    "feature_encoder.row_cls_embedding.": "row_cls.",
    "feature_encoder.col_cls_embedding.": "col_cls.",
    "decoder.layers.": "decoder.",
}

_ATTENTION = {
    "self_attention_between_features": "row_attn",
    "self_attention_between_datapoints": "col_attn",
}

_BLOCK = {
    "norm1": "row_norm",
    "norm2": "col_norm",
    "norm3": "mlp_norm",
    "ff_gate": "mlp.gate_lin",
    "ff_value": "mlp.up_lin",
    "ff_out": "mlp.down_lin",
    "feature_residual_scale": "row_gain",
    "datapoint_residual_scale": "col_gain",
    "ffn_residual_scale": "mlp_gain",
}


def remap_ckpt(ckpt: dict[str, Tensor]) -> dict[str, Tensor]:
    r"""Map a released MatrixFormer expert checkpoint to module names."""
    out: dict[str, Tensor] = {}
    qkv: dict[str, dict[str, Tensor]] = {}

    for key, value in ckpt.items():
        if key == "borders":
            out[key] = value
        elif key == "feature_encoder.mask_token":
            out["mask_token"] = value.view(-1)
        elif key.startswith(tuple(_ENCODER)):
            prefix = next(p for p in _ENCODER if key.startswith(p))
            out[_ENCODER[prefix] + key.removeprefix(prefix)] = value
        elif key.startswith("transformer_blocks."):
            layer, _, rest = key.removeprefix("transformer_blocks.").partition(
                "."
            )
            name, _, tail = rest.partition(".")
            prefix = f"blocks.{layer}."
            if name in _ATTENTION:
                attn = prefix + _ATTENTION[name]
                proj, param = tail.split(".", 1)
                if proj == "rope":
                    out[f"{attn}.query_transform.inv_freq"] = value
                    out[f"{attn}.key_transform.inv_freq"] = value
                elif proj == "out_proj":
                    out[f"{attn}.out_lin.{param}"] = value
                else:
                    qkv.setdefault(f"{attn}.qkv_lin.{param}", {})[proj] = value
            else:
                out[prefix + _BLOCK[name] + (f".{tail}" if tail else "")] = (
                    value
                )
        else:
            raise KeyError(f"Unexpected MatrixFormer checkpoint key {key!r}")

    for key, parts in qkv.items():
        out[key] = torch.cat(
            [parts["q_proj"], parts["k_proj"], parts["v_proj"]]
        )

    return out
