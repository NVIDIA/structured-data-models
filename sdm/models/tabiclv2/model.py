# ruff: noqa: D205

import hashlib
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import Model
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.recipe import default_recipe
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.processing import Recipe

_CHECKPOINT_REPO_ID = "jingang/TabICL"
_CHECKPOINT_FILENAMES = {
    "classifier": "tabicl-classifier-v2-20260212.ckpt",
    "regressor": "tabicl-regressor-v2-20260212.ckpt",
}


class TabICLv2(Model):
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
        batch_size_limit: Maximum number of independent feature or row
            attention batches processed at once. ``None`` disables attention
            batch chunking.
    """

    #:
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
        batch_size_limit: int | None = None,
    ) -> None:
        super().__init__()

        if batch_size_limit is not None and (
            isinstance(batch_size_limit, bool)
            or not isinstance(batch_size_limit, int)
            or batch_size_limit < 1
        ):
            raise ValueError(
                "batch_size_limit must be a positive integer or None."
            )
        self.batch_size_limit = batch_size_limit

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

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def load_regression_checkpoint(
        self,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
    ) -> "TabICLv2":
        r"""Load and verify a local regression checkpoint.

        Unlike ``pretrained=True``, this method never resolves an artifact
        through Hugging Face. It is intended for reproducible benchmark runs
        that pin the regression checkpoint and its SHA-256 digest.

        Args:
            checkpoint_path: Path to a local published regression checkpoint.
            checkpoint_sha256: Expected SHA-256 digest of the checkpoint.

        Returns:
            This model with its regression network loaded.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not name a file.
            ValueError: If the digest is invalid or does not match the local
                checkpoint.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file does not exist: '{path}'."
            )

        expected_hash = _validate_sha256(checkpoint_sha256)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for checkpoint '{path}': expected "
                f"{expected_hash}, got {actual_hash}."
            )

        self._load_checkpoint(path, is_classifier=False)
        return self

    def _load_from_pretrained(self) -> "TabICLv2":
        for variant, is_classifier in [
            ("classifier", True),
            ("regressor", False),
        ]:
            try:
                path = hf_hub_download(
                    repo_id=_CHECKPOINT_REPO_ID,
                    filename=_CHECKPOINT_FILENAMES[variant],
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                path = hf_hub_download(
                    repo_id=_CHECKPOINT_REPO_ID,
                    filename=_CHECKPOINT_FILENAMES[variant],
                )
            self._load_checkpoint(path, is_classifier=is_classifier)

        return self

    def _load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        is_classifier: bool,
    ) -> None:
        device = next(self.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            state_dict = checkpoint["state_dict"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Expected checkpoint to contain a 'state_dict' mapping."
            ) from error

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Expected checkpoint 'state_dict' to be a dictionary."
            )

        state_dict = _remap_ckpt(
            state_dict,
            is_classifier=is_classifier,
        )
        model = self.cls_model if is_classifier else self.reg_model
        model.load_state_dict(state_dict, strict=True)

    def _forward(  # TODO Add multi-class support.
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> TableTensor:  # [..., R_query, num_classes or 999]

        if x_context is None and x_query is not None:
            x = x_query.numerical
        elif x_query is None and x_context is not None:
            x = x_context.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        y: Tensor | None = None
        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.as_tensor().squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        elif cache is not None:
            classes = cast(Tensor, cache["classes"])

        if y is None:
            y = x.new_empty(
                (*x.size()[:2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if classes is None:
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=self.reg_model(
                    x,
                    y,
                    cache=cache,
                    batch_size_limit=self.batch_size_limit,
                ).sort(dim=-1)[0],
            )

        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=self.cls_model(
                x,
                y,
                cache=cache,
                batch_size_limit=self.batch_size_limit,
            )[..., : len(classes)],
        )

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
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        x = self.row_embedding(
            x=x,
            y=y,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
        x = self.icl_block(
            x=x,
            y=y,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )
        return self.head(x)


# Helpers #####################################################################


def _validate_sha256(checkpoint_sha256: str) -> str:
    if not isinstance(checkpoint_sha256, str):
        raise ValueError(
            "Expected 'checkpoint_sha256' to be a SHA-256 string."
        )

    normalized = checkpoint_sha256.strip().lower()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(
            "Expected 'checkpoint_sha256' to be a 64-character hexadecimal "
            "SHA-256 digest."
        )
    return normalized


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: '{path}'.")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
