# ruff: noqa: D205

from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding


class TabICLv2(torch.nn.Module):
    r"""The tabular foundation model from the `"TabICLv2: A Better, Faster,
    Scalable, and Open Tabular Foundation Model"
    <https://arxiv.org/abs/2602.11139>`_ paper.

    .. image:: https://arxiv.org/html/2602.11139v1/x2.png
        :align: center
        :width: 600px

    :class:`TabICLv2` treats a table as an in-context learning problem: a set
    of labeled training rows provides context, and the model predicts targets
    for held-out test rows from the same table.

    Feature columns are encoded via repeated feature grouping, where each
    feature participates in multiple shifted feature groups.
    This breaks symmetries between similarly distributed columns while
    preserving fine-grained feature information.
    Target-aware embeddings are then added to the training-row feature
    representations, injecting label information early without giving test rows
    access to their own targets.

    Afterwards, the module applies three attention stages in sequence:

    * **Column-wise:** Each grouped feature is processed as a set of row tokens
      with induced set attention.
      The inducing tokens summarize information from the in-context training
      rows, then pass it back to the row tokens, giving each feature group a
      target-aware representation of its values across examples.
      Query-Aware Scalable SoftMax (:class:`~sdm.nn.QASSMax`) sharpens
      attention over long contexts and reduces attention fading as the number
      of rows grows.
    * **Row-wise:** For each row, the feature-group embeddings are processed
      together with learnable readout tokens.
      Attention across the grouped features lets the model combine column
      evidence and feature interactions within that row.
      The readout token outputs are concatenated to form a fixed-size row
      embedding.
    * **Dataset-wise:** The row embeddings are processed across the dataset
      for in-context prediction.
      Training-row embeddings are combined with target embeddings, and test
      rows attend to the labeled training rows. The resulting test-row states
      are mapped to task outputs, such as class logits for classification or
      quantile predictions for regression.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.cls_model = _TabICLv2(
            num_classes=10,
            num_quantiles=0,
            norm_bias=True,
            device=device,
        )
        self.reg_model = _TabICLv2(
            num_classes=0,
            num_quantiles=999,
            norm_bias=False,
            device=device,
        )

        if pretrained:
            self._load_from_pretrained()

    def _load_from_pretrained(self) -> "TabICLv2":
        device = next(self.parameters()).device

        for variant in ["classifier", "regressor"]:
            try:
                path = hf_hub_download(
                    repo_id="jingang/TabICL",
                    filename=f"tabicl-{variant}-v2-20260212.ckpt",
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                path = hf_hub_download(
                    repo_id="jingang/TabICL",
                    filename=f"tabicl-{variant}-v2-20260212.ckpt",
                )
            ckpt = torch.load(path, map_location=device)["state_dict"]

            if variant == "classifier":
                ckpt = _remap_ckpt(ckpt, is_classifier=True)
                self.cls_model.load_state_dict(ckpt)
            else:
                ckpt = _remap_ckpt(ckpt, is_classifier=False)
                self.reg_model.load_state_dict(ckpt)

        return self

    @torch.inference_mode()
    def forward(  # TODO Add multi-class support.
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, num_classes or 999]
        r"""The forward pass.

        Args:
            x: The feature tensor with shape ``[B, R, C]`` for ``B`` tables,
                ``R`` rows, and ``C`` columns.
                The first ``R_train`` rows along ``R`` refer to the in-context
                examples.
            y: The targets of in-context examples with shape ``[B, R_train]``.
                Integer ``y`` refer to classification tasks.
                Floating-point ``y`` refer to regression tasks.

        Returns:
            Tensor with shape ``[B, R_test, num_classes]`` for integer ``y``
            and ``[B, R_test, 999]`` for floating-point ``y``.
            Integer ``y`` return class logits.
            Floating-point ``y`` return 999 quantiles at probability levels
            :math:`\left\{0.001, 0.002, \ldots, 0.999\right\}`.
        """
        if y.is_floating_point():
            return self.reg_model(x, y)
        return self.cls_model(x, y)

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"


class _TabICLv2(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        channels: int = 128,
        num_embedding_layers: int = 3,
        num_embedding_heads: int = 8,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_readout_tokens: int = 4,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        norm_bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.head = Sequential(
            Linear(
                in_features=num_readout_tokens * channels,
                out_features=2 * num_readout_tokens * channels,
                **factory_kwargs,
            ),
            GELU(),
            Linear(
                in_features=2 * num_readout_tokens * channels,
                out_features=num_classes or num_quantiles,
                **factory_kwargs,
            ),
        )

    def forward(
        self,
        x: Tensor,  # [B, R, C]
        y: Tensor,  # [B, R_train]
    ) -> Tensor:  # [B, R_test, num_classes or num_quantiles]
        x = self.row_embedding(x, y)
        x = self.icl_block(x, y)
        return self.head(x)


# Helpers #####################################################################


def _remap_ckpt(
    ckpt: dict[str, Tensor],
    is_classifier: bool,
) -> dict[str, Tensor]:

    def _map_transformer(prefix: str, tail: str) -> list[str]:
        tail = tail.replace("attn.in_proj_weight", "attn.qkv_lin.weight")
        tail = tail.replace("attn.in_proj_bias", "attn.qkv_lin.bias")
        tail = tail.replace("attn.out_proj.", "attn.out_lin.")
        tail = tail.replace(
            "attn.ssmax_layer.base_mlp.",
            "attn.sdpa.qassmax.scale.",
        )
        tail = tail.replace(
            "attn.ssmax_layer.query_mlp.",
            "attn.sdpa.qassmax.gate.",
        )

        if tail.startswith("norm1."):
            return [
                prefix + tail.replace("norm1.", "q_norm.", 1),
                prefix + tail.replace("norm1.", "kv_norm.", 1),
            ]
        if tail.startswith("norm2."):
            tail = tail.replace("norm2.", "mlp.0.", 1)
        elif tail.startswith("linear1."):
            tail = tail.replace("linear1.", "mlp.1.", 1)
        elif tail.startswith("linear2."):
            tail = tail.replace("linear2.", "mlp.3.", 1)

        return [prefix + tail]

    out: dict[str, Tensor] = {}

    if is_classifier:
        out["row_embedding.y_emb.weight"] = (
            ckpt["col_embedder.y_encoder.weight"].t()
            + ckpt["col_embedder.y_encoder.bias"]
        )
        out["icl_block.y_emb.weight"] = (
            ckpt["icl_predictor.y_encoder.weight"].t()
            + ckpt["icl_predictor.y_encoder.bias"]
        )
    else:
        out["row_embedding.y_lin.weight"] = ckpt[
            "col_embedder.y_encoder.weight"
        ]
        out["row_embedding.y_lin.bias"] = ckpt["col_embedder.y_encoder.bias"]
        out["icl_block.y_lin.weight"] = ckpt["icl_predictor.y_encoder.weight"]
        out["icl_block.y_lin.bias"] = ckpt["icl_predictor.y_encoder.bias"]

    for key, value in ckpt.items():
        if key.startswith("col_embedder.in_linear."):
            new_key = key.replace(
                "col_embedder.in_linear",
                "row_embedding.lin",
            )
            out[new_key] = value

        elif key.startswith("col_embedder.tf_col.blocks."):
            layer, tail = key.removeprefix(
                "col_embedder.tf_col.blocks."
            ).split(".", 1)

            if tail == "ind_vectors":
                out[f"row_embedding.col_layers.{layer}.inducing_points"] = (
                    value
                )
            elif tail.startswith("multihead_attn1."):
                prefix = f"row_embedding.col_layers.{layer}.transformer_1."
                for new_key in _map_transformer(
                    prefix,
                    tail.removeprefix("multihead_attn1."),
                ):
                    out[new_key] = value
            elif tail.startswith("multihead_attn2."):
                prefix = f"row_embedding.col_layers.{layer}.transformer_2."
                for new_key in _map_transformer(
                    prefix,
                    tail.removeprefix("multihead_attn2."),
                ):
                    out[new_key] = value

        elif key == "row_interactor.cls_tokens":
            value = value.unsqueeze(0).unsqueeze(0)
            out["row_embedding.readout_token"] = value

        elif key.startswith("row_interactor.tf_row.blocks."):
            layer, tail = key.removeprefix(
                "row_interactor.tf_row.blocks."
            ).split(".", 1)
            prefix = f"row_embedding.row_layers.{layer}."
            for new_key in _map_transformer(prefix, tail):
                out[new_key] = value

        elif key == "row_interactor.tf_row.rope.freqs":
            out["row_embedding.rope.inv_freq"] = value

        elif key.startswith("row_interactor.out_ln."):
            out[key.replace("row_interactor.out_ln", "row_embedding.norm")] = (
                value
            )

        elif key.startswith("icl_predictor.tf_icl.blocks."):
            layer, tail = key.removeprefix(
                "icl_predictor.tf_icl.blocks."
            ).split(".", 1)
            prefix = f"icl_block.layers.{layer}."
            for new_key in _map_transformer(prefix, tail):
                out[new_key] = value

        elif key.startswith("icl_predictor.ln."):
            out[key.replace("icl_predictor.ln", "icl_block.norm")] = value

        elif key.startswith("icl_predictor.decoder."):
            out[key.replace("icl_predictor.decoder", "head")] = value

    return out
