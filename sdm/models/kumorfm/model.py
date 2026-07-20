# ruff: noqa: D205
from typing import Any, ClassVar, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.recipe import default_recipe
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.processing import Recipe


class KumoRFM(ICLModel):
    r"""The adapted relational foundation model from the `"KumoRFM-2: Scaling
    Foundation Models for Relational Learning"
    <https://arxiv.org/abs/2604.12596>`_ paper.

    .. image:: https://arxiv.org/html/2604.12596v1/x3.png
        :align: center
        :width: 100%

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
    """

    #:
    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.datetime}
    )
    #:
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    #:
    supports_related_tables: ClassVar[bool] = True

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.cls_model = _KumoRFM(
            num_classes=10,
            num_quantiles=0,
            norm_bias=True,
            device=device,
        )
        self.reg_model = _KumoRFM(
            num_classes=0,
            num_quantiles=999,
            norm_bias=False,
            device=device,
        )

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, *]

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            classes = y_context.categorical.categories[0]
        elif cache is not None:
            classes = cast(Tensor | None, cache["classes"])

        if classes is None:
            out = self.reg_model(
                x_context=x_context,
                y_context=y_context,
                x_query=x_query,
                related_context_tables=related_context_tables,
                related_query_tables=related_query_tables,
                cache=cache,
                num_hops=kwargs.get("num_hops"),
            )
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out.sort(dim=-1)[0],
            )

        out = self.cls_model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            cache=cache,
            num_classes=len(classes),
            num_hops=kwargs.get("num_hops"),
        )
        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out[..., : len(classes)],
        )

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()


class _KumoRFM(torch.nn.Module):
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
        max_train_size: int = 20_000,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_classes = num_classes
        self.max_train_size = max_train_size

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
        self.gnn = InvariantGNN(
            channels=num_readout_tokens * channels,
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
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        *,
        cache: Cache | None = None,
        num_classes: int | None = None,
        num_hops: int | None = None,
    ) -> Tensor:  # [..., R_query, *]

        # TODO Support `fit+predict`:
        assert x_context is not None
        assert y_context is not None
        assert x_query is not None
        assert related_context_tables is not None
        assert related_query_tables is not None

        if num_hops is None:
            num_hops = 2  # TODO Support automatic `num_hops` detection.
            # TODO Write to `cache` if available.

        # TODO Assert `len(task_links) == 1`>
        # TODO Support computing relative time.
        # TODO Inject task-features.
        # TODO Inject random heterogeneous GNN.

        xs_context: dict[str, Tensor] = {}
        xs_query: dict[str, Tensor] = {}
        # TODO Apply per hop.
        for name in related_context_tables.tables:
            x_context_i = related_context_tables.tables[name].numerical
            x_query_i = related_query_tables.tables[name].numerical
            xs_context[name], xs_query[name] = self.row_embedding(
                x=torch.cat([x_context_i, x_query_i], dim=-2),
                y=torch.randint(  # TODO Inject real label.
                    low=0,
                    high=2,
                    size=x_context_i.size()[:-1],
                    dtype=torch.int64
                    if y_context.categorical.size(-1) > 0
                    else torch.float32,
                    device=y_context.device,
                ),
                max_keys=self.max_train_size,
                num_classes=num_classes,
                cache=None,  # TODO
                generator=None,  # TODO
            ).split([x_context_i.size(-2), x_query_i.size(-2)], dim=-2)

        x_context = self.gnn(
            x=torch.cat(
                [xs_context[name] for name in related_context_tables.tables],
                dim=-2,
            ),
            graph=HomogeneousGraph.from_related_tables(related_context_tables),
            readout_table=related_context_tables.task_links[0].table,
            num_hops=num_hops,
            generator=None,  # TODO
        )
        x_query = self.gnn(  # TODO Make sure we use same edge type embeddings!
            x=torch.cat(
                [xs_query[name] for name in related_query_tables.tables],
                dim=-2,
            ),
            graph=HomogeneousGraph.from_related_tables(related_query_tables),
            readout_table=related_query_tables.task_links[0].table,
            num_hops=num_hops,
            generator=None,  # TODO
        )

        x = torch.cat([x_context, x_query], dim=-2)
        if y_context.categorical.size(-1) > 0:
            y = y_context.categorical.as_tensor().squeeze(-1)
        elif y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)

        x = self.icl_block(x, y)
        return self.head(x)
