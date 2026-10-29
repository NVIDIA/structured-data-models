# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import torch
from safetensors.torch import load_file
from torch import Tensor

import sdm.processing as sp
from sdm import (
    Recipe,
    RelatedTables,
    Stype,
    TableTensor,
    Task,
)
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.timestral.config import LAYER_KIND_AXES, _TimestralConfig
from sdm.models.timestral.decode import _Decoder, _ffill_with_mask
from sdm.models.timestral.encoding import _encode_categories

_LEVELS = (0.01, *[i / 100 for i in range(5, 100, 5)], 0.99)


class Timestral(ICLModel):
    r"""The time series forecasting foundation model from the
    `"Timestral" <https://arxiv.org/abs/2610.37585>`__ paper.

    Args:
        size: The model size, one of ``"small"``, ``"medium"``, or
            ``"large"``. Defaults to ``"large"``.
        pretrained: Whether to load pretrained checkpoints.
        accept_license: Whether to accept the `Modified MIT License
            <https://huggingface.co/mistralai/Timestral-209M/blob/main/
            LICENSE>`__ without showing the interactive license prompt.
        device: The device for model parameters. If ``None``, uses PyTorch's
            default device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_multi_target: ClassVar[bool] = True
    supports_context_only_columns: ClassVar[bool] = True
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        size: Literal["tiny", "small", "medium", "large"] = "large",
        pretrained: bool = True,
        accept_license: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)

        self.model = _Timestral(device="meta" if pretrained else device)

        if pretrained:
            self._load_from_pretrained(size, accept_license, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return Recipe()

    def _load_from_pretrained(
        self,
        size: Literal["tiny", "small", "medium", "large"],
        device: torch.device | str | None,
    ) -> Timestral:
        device = torch.get_default_device() if device is None else device

        model_size, revision = {
                "tiny": ("5M", "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"),
                "small": ("28M", "180698bbf5a50dda3516aa6419798437fb03a430"),
                "medium": ("123M", "5a147287addda85f6acd8da3fb8858fe9af7e8f7"),
                "large": ("209M", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"),
        }[size]

            config_path = download_checkpoint(
                repo_id= f"mistralai/Timestral-{model_size}",
                filename="config.json",
                revision=revision,
            )
            config = _TimestralConfig(
                **json.loads(Path(config_path).read_text())
            )
        else:
            # Released dimensions for offline initialization.
            dims = {
                "5M": (192, 3, 512, 8),
                "28M": (384, 6, 1024, 12),
                "123M": (640, 10, 1712, 20),
                "209M": (768, 12, 2048, 24),
            }
            width, heads, hidden, depth = dims[model_size]
            config = _TimestralConfig(
                patch_size=32,
                d_model=width,
                num_heads=heads,
                mlp_hidden_dim=hidden,
                block_schedule=tuple(
                    "time_variate" if i % 4 == 3 else "time"
                    for i in range(depth)
                ),
                quantile_knots=_LEVELS,
                block_size=32,
                context_length=32768,
                quantile_width_factor=3.0,
            )
        with torch.device(device or torch.get_default_device()):
            self.model = _TimestralModel(config)
        if pretrained:
            path = download_checkpoint(
                repo_id=repo,
                filename="model.safetensors",
                revision=revision,
                license_prompt=None
                if accept_license
                else (
                    "Timestral weights use Modified MIT terms, including a "
                    "company revenue restriction. Review "
                    f"https://huggingface.co/{repo}/blob/{revision}/LICENSE"
                ),
            )
            state = load_file(
                path, device=str(next(self.model.parameters()).device)
            )
            state = _remap_checkpoint(state, config)
            for key, value in self.model.state_dict().items():
                if key.endswith(".inv_freq"):
                    state[key] = value.detach()
            self.model.load_state_dict(state, strict=True)
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:  # noqa: D102
        return Recipe(
            features=sp.AlignCategories(min_frequency=0, unknown="append"),
            output=sp.AverageEstimators(),
        )

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
        **kwargs: Any,
    ) -> TableTensor:
        if cache is not None and not cache.is_recording:
            x_context = cast(TableTensor, cache["context"])
            y_context = cast(TableTensor, cache["target"])
            x_context_only = cast(TableTensor | None, cache["past"])
        assert x_context is not None
        assert y_context is not None
        if cache is not None and cache.is_recording:
            cache["context"] = x_context
            cache["target"] = y_context
            cache["past"] = x_context_only
        names = y_context.columns[Stype.numerical]
        columns = [
            f"{name}__q{round(q * 100)}"
            for name in names
            for q in self.model.config.quantile_knots
        ]
        if x_query is None or x_query.size(-2) == 0:
            return TableTensor.from_tensor(
                y_context.numerical.new_empty(
                    (*y_context.shape[:-2], 0, len(columns))
                ),
                columns=columns,
            )
        y = y_context.numerical.transpose(-2, -1)
        batch = y.shape[:-2]
        y = y.reshape(-1, *y.shape[-2:]).float()
        observed = y.isfinite()
        y = y.masked_fill(~observed, 0)
        decoder = _Decoder(self.model)
        past = self._covariates(
            context=x_context_only,
            query=None,
            target=y,
            observed=observed,
            max_context=decoder.max_context,
        )
        known = self._covariates(
            context=x_context,
            query=x_query,
            target=y,
            observed=observed,
            max_context=decoder.max_context,
        )
        horizon = x_query.size(-2)
        out = decoder._forecast_raw(y, observed, horizon, past, known)
        if symmetric_averaging:
            mirror = decoder._forecast_raw(
                inputs=-y,
                observation_mask=observed,
                prediction_length=horizon,
                past_covariates=-past if past is not None else None,
                known_covariates=-known if known is not None else None,
            )
            out = (out - mirror.flip(-1)) * 0.5
        if clip_nonnegative:
            eligible = observed.any(-1) & ((y >= 0) | ~observed).all(-1)
            out = torch.where(eligible[..., None, None], out.clamp_min(0), out)
        out = out.transpose(-3, -2).reshape(*batch, horizon, len(columns))
        return TableTensor.from_tensor(out, columns=columns)

    @staticmethod
    def _covariates(
        context: TableTensor | None,
        query: TableTensor | None,
        target: Tensor,
        observed: Tensor,
        max_context: int,
    ) -> Tensor | None:
        if context is None or context.size(-1) == 0:
            return None
        numeric = context.numerical
        codes = context.categorical.code
        num_categories = max(
            (c.numel() for c in context.categorical.categories), default=0
        )
        if query is not None:
            numeric = torch.cat([numeric, query.numerical], dim=-2)
            codes = torch.cat([codes, query.categorical.code], dim=-2)
            num_categories = max(
                num_categories,
                max(
                    (c.numel() for c in query.categorical.categories),
                    default=0,
                ),
            )
        length = numeric.size(-2)
        numeric = (
            numeric.transpose(-2, -1)
            .reshape(target.size(0), -1, length)
            .float()
        )
        codes = codes.transpose(-2, -1).reshape(target.size(0), -1, length)
        if codes.size(1) == 0:
            return numeric
        if target.size(1) == 1:
            started = observed[:, 0].cummax(-1).values
            filled = _ffill_with_mask(target, observed)[:, 0].masked_fill(
                ~started, torch.nan
            )
            categorical = _encode_categories(
                target=filled,
                codes=codes,
                max_context=max_context,
                num_categories=num_categories,
            )
        else:
            categorical = codes.float().masked_fill(codes < 0, torch.nan)
        return torch.cat([numeric, categorical], dim=1)


class _TimestralModel(torch.nn.Module):
    def __init__(self, config: _TimestralConfig) -> None:
        super().__init__()
        self.config = config


def _remap_checkpoint(
    state: dict[str, Tensor], config: _TimestralConfig
) -> dict[str, Tensor]:
    out = {}
    heads = config.num_heads
    time_attns = {
        f"transformer.layers.{layer}.attns.{axis}"
        for layer, kind in enumerate(config.block_schedule)
        for axis, is_time in enumerate(LAYER_KIND_AXES[kind])
        if is_time
    }
    channels = torch.arange(
        config.head_dim,
        device=next(iter(state.values())).device,
    )
    half, rotated_half = config.head_dim // 2, config.rope_dim // 2
    # Move the active pairs from the full head halves into a contiguous prefix.
    permutation = torch.cat(
        [
            channels[:rotated_half],
            channels[half : half + rotated_half],
            channels[rotated_half:half],
            channels[half + rotated_half :],
        ]
    )
    column_every = config.column_every
    for key, value in state.items():
        key = (
            key.replace("hidden_layer", "mlp.0")
            .replace("output_layer", "mlp.2")
            .replace("residual_layer", "res")
        )
        if ".wQKV." in key:
            # Upstream interleaves heads within each head channel.
            head_dim = value.size(0) // (3 * heads)
            trailing = value.shape[1:]
            qkv = value.reshape(3, head_dim, heads, *trailing).transpose(1, 2)
            if key.rsplit(".", 2)[0] in time_attns:
                qkv = torch.cat(
                    [
                        qkv[:2].index_select(2, permutation),
                        qkv[2:],
                    ]
                )
            value = qkv.reshape_as(value)
            key = key.replace("wQKV", "qkv_lin")
        if key.startswith("transformer.layers."):
            parts = key.split(".")
            layer = int(parts[2])
            if parts[3] in {"attns", "attn_norms"}:
                column = int(parts[4]) == 1
                module = "attn" if parts[3] == "attns" else "query_norm"
                suffix = ".".join(parts[5:])
            else:
                column = (layer + 1) % column_every == 0
                module = "mlp.0" if parts[3] == "mlp_norm" else "mlp.1"
                suffix = ".".join(parts[4:])
            stack = "col_blocks" if column else "row_blocks"
            index = layer // column_every if column else layer
            key = f"transformer.{stack}.{index}.{module}.{suffix}"
        key = key.replace(".wO.", ".out_lin.").replace("._gate.", ".gate.0.")
        key = key.replace("transformer.out_norm.", "transformer.norm.")
        if key.endswith(".scale"):
            key = key[:-5] + "weight"
        if ".query_norm." in key:
            out[key.replace(".query_norm.", ".key_value_norm.")] = value
        if ".mlp.1.w12." in key:
            # Upstream stores the linear branch first, the SiLU branch second.
            up, gate = value.chunk(2, dim=0)
            out[key.replace("w12", "up_lin")] = up
            out[key.replace("w12", "gate_lin")] = gate
        else:
            out[key.replace(".mlp.1.w3.", ".mlp.1.down_lin.")] = value
    return out
