# ruff: noqa: D205

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.tabiclv2.ckpt import remap_ckpt
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.recipe import default_recipe
from sdm.models.tabiclv2.row_embedding import RowEmbedding


class TabICLv2(ICLModel):
    r"""The tabular foundation model from the `"TabICLv2: A Better, Faster,
    Scalable, and Open Tabular Foundation Model"
    <https://arxiv.org/abs/2602.11139>`_ paper.

    .. figure:: /images/tabicl_light.svg
        :figclass: light-only
        :align: center
        :width: 600px

    .. figure:: /images/tabicl_dark.svg
        :figclass: dark-only
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

    .. testcode::

        from sdm import TableTensor
        from sdm.models import TabICLv2

        table = TableTensor.from_columns(
            {
                "col0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "col1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "target": ["t", "f", "t", "f", "t", None, None, None],
            },
            stypes={
                "col0": "numerical",
                "col1": "numerical",
                "target": "categorical",
            },
            device="cuda",
        )
        model = TabICLv2(device="cuda")

        # Default in-context learning forward pass:
        out = model(
            x_context=table[:5].drop_columns("target"),
            y_context=table[:5, "target"],
            x_query=table[5:].drop_columns("target"),
        )
        assert out.size() == (3, 2)

        # Fit+Predict forward pass via key/value caching:
        model.fit(
            x=table[:5].drop_columns("target"),
            y=table[:5, "target"],
        )
        out = model.predict(table[5:].drop_columns("target"))
        assert out.size() == (3, 2)

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=None)

        self.cls_model = _TabICLv2(
            num_classes=10,
            num_quantiles=0,
            norm_bias=True,
            device="meta" if pretrained else device,
        )
        self.reg_model = _TabICLv2(
            num_classes=0,
            num_quantiles=999,
            norm_bias=False,
            device="meta" if pretrained else device,
        )

        if pretrained:
            self._load_from_pretrained(device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        device: torch.device | str | None,
    ) -> TabICLv2:
        device = torch.get_default_device() if device is None else device

        for variant in ["classifier", "regressor"]:
            path = download_checkpoint(
                repo_id="jingang/TabICL",
                filename=f"tabicl-{variant}-v2-20260212.ckpt",
            )
            ckpt = torch.load(
                path,
                map_location=device,
                weights_only=True,
            )["state_dict"]

            if variant == "classifier":
                ckpt = remap_ckpt(ckpt, is_classifier=True)
                self.cls_model.load_state_dict(ckpt, assign=True)
            else:
                ckpt = remap_ckpt(ckpt, is_classifier=False)
                self.reg_model.load_state_dict(ckpt, assign=True)

        return self

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 999]

        if x_query is None and x_context is not None:
            x = x_context.numerical
        elif x_context is None and x_query is not None:
            x = x_query.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        y: Tensor | None = None
        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            classes = cast(Tensor | None, cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if classes is None:
            out = self.reg_model(x, y, cache=cache)
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out.sort(dim=-1)[0],
            )

        out = self.cls_model(x, y, cache=cache, num_classes=len(classes))
        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out[..., : len(classes)],
        )


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
            out_channels=num_classes or num_quantiles,
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            norm_bias=norm_bias,
            temperature=0.9,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
        num_classes: int | None = None,
    ) -> Tensor:  # [..., R_test, out_channels or num_classes]
        if not y.is_floating_point():
            assert num_classes is not None

        x = self.row_embedding(x, y, num_classes=num_classes, cache=cache)
        return self.icl_block(
            x=x,
            y=y,
            num_classes=num_classes,
            cache=cache,
        )
