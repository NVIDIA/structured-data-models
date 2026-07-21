# ruff: noqa: D205
from typing import Any, ClassVar, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
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

    _checkpoint_filenames: ClassVar[dict[str, str]] = {
        "classifier": "cls-model.pt",
        "regressor": "reg-model.pt",
    }

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
            norm_bias=True,
            device=device,
        )

        if pretrained:
            self._load_from_pretrained()

        self.eval()

    def _load_from_pretrained(self) -> "KumoRFM":
        device = next(self.parameters()).device

        for variant, filename in self._checkpoint_filenames.items():
            path = download_checkpoint(
                repo_id="nvidia/kumorfm-2",
                filename=filename,
                revision="v2.1.0",
            )
            checkpoint = torch.load(
                path,
                map_location=device,
                weights_only=True,
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            model = (
                self.cls_model if variant == "classifier" else self.reg_model
            )
            model.load_state_dict(
                _remap_v2_1_checkpoint(
                    state_dict,
                    is_classifier=variant == "classifier",
                ),
                strict=True,
            )

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
    ) -> TableTensor:  # [..., R_query, *]

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            classes = y_context.categorical.categories[0]
        elif cache is not None:
            classes = cast(Tensor | None, cache["classes"])

        out = (self.reg_model if classes is None else self.cls_model)(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            cache=cache,
            generator=generator,
            num_hops=kwargs.get("num_hops"),
        )

        if classes is None:
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out.sort(dim=-1)[0],
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
        generator: torch.Generator | None = None,
        num_hops: int | None = None,
    ) -> Tensor:  # [..., R_query, *]

        if related_context_tables is None and related_query_tables is None:
            raise ValueError(
                f"'{self.__class__.__name__}' requires related tables"
            )

        for tables in (related_context_tables, related_query_tables):
            if tables is not None and len(tables.task_links) != 1:
                raise NotImplementedError(
                    f"'{self.__class__.__name__}' expects exactly one task "
                    f"link to an entity table (got {len(tables.task_links)})"
                )

        if num_hops is None:  # Infer `num_hops`:
            num_hops = 2  # TODO Support automatic `num_hops` detection.
            if cache is not None and cache.is_recording:
                cast(dict[str, Any], cache["kwargs"])["num_hops"] = num_hops

        num_classes: int | None = None  # Extract `y` as tensor:
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.as_tensor().squeeze(-1)
            num_classes = len(y_context.categorical.categories[0])
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            if isinstance(cache["classes"], Tensor):
                num_classes = len(cache["classes"])
            y = torch.empty(
                (0,),  # NOTE Guaranteed to be 1D for now.
                dtype=torch.float32 if num_classes is None else torch.int64,
                device=next(self.parameters()).device,
            )

        # TODO Support computing relative time.
        # TODO Inject task-features.
        # TODO Inject random heterogeneous GNN.
        # TODO Make sure edge types are aligned!

        # Reason within each Table ############################################
        xs_context: dict[str, Tensor] = {}
        xs_query: dict[str, Tensor] = {}
        for name in cast(
            RelatedTables,
            related_context_tables or related_query_tables,
        ).tables:
            table_cache: Cache | None = None
            if cache is not None and cache.is_recording:
                table_cache = Cache()
            elif cache is not None:
                table_cache = cast(Cache, cache[f"table_{name}"])

            y_i = y  # Distribute `target` over related tables:
            if related_context_tables is not None:
                # TODO Split entity table into task+nearby entities.
                x_i = related_context_tables.tables[name].numerical
                y_i = y_i[:1].expand(x_i.size(-2))  # TODO
                if related_query_tables is not None:
                    x_i = torch.cat(
                        [x_i, related_query_tables.tables[name].numerical],
                        dim=-2,
                    )
            else:
                assert related_query_tables is not None
                x_i = related_query_tables.tables[name].numerical

            x_i = self.row_embedding(
                x=x_i,
                y=y_i,
                max_keys=self.max_train_size,
                num_classes=num_classes,
                cache=table_cache,
                generator=generator,
            )

            if related_context_tables is not None:
                num_rows = related_context_tables.tables[name].size(-2)
                xs_context[name] = x_i[..., :num_rows, :]
            if related_query_tables is not None:
                num_rows = related_query_tables.tables[name].size(-2)
                xs_query[name] = x_i[..., -num_rows:, :]

            if cache is not None and cache.is_recording:
                cache[f"table_{name}"] = cast(Cache, table_cache)

        # Inter-Message Passing Exchange ######################################
        edge_type_emb: Tensor | None = None
        if related_context_tables is not None:
            x_context: Tensor = torch.cat(
                [xs_context[name] for name in related_context_tables.tables],
                dim=-2,
            )
            del xs_context
            graph = HomogeneousGraph.from_tables(related_context_tables)
            edge_type_emb = self.gnn.get_edge_type_emb(
                num_edge_types=graph.num_edge_types,
                dtype=x_context.dtype,
                generator=generator,
            )
            if cache is not None and cache.is_recording:
                cache["edge_type_emb"] = edge_type_emb
            x_context = self.gnn(
                x=x_context,
                graph=graph,
                edge_type_emb=edge_type_emb,
                readout_table=related_context_tables.task_links[0].table,
                num_hops=num_hops,
            )

        if related_query_tables is not None:
            x_query: Tensor = torch.cat(
                [xs_query[name] for name in related_query_tables.tables],
                dim=-2,
            )
            del xs_query
            graph = HomogeneousGraph.from_tables(related_query_tables)
            if edge_type_emb is None:
                assert cache is not None
                edge_type_emb = cast(Tensor, cache["edge_type_emb"])
            x_query = self.gnn(
                x=x_query,
                graph=graph,
                edge_type_emb=edge_type_emb,
                readout_table=related_query_tables.task_links[0].table,
                num_hops=num_hops,
            )

        # Reason across Tables ################################################
        if x_query is None and x_context is not None:
            x = x_context
        elif x_context is None and x_query is not None:
            x = x_query
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context, x_query], dim=-2)
            del x_context
            del x_query
        x = self.icl_block(x, y, cache=cache)
        return self.head(x)


def _remap_v2_1_checkpoint(
    state_dict: dict[str, Tensor],
    *,
    is_classifier: bool,
) -> dict[str, Tensor]:
    def _map_transformer(tail: str) -> str:
        replacements = {
            "norm1_1": "q_norm",
            "norm1_2": "kv_norm",
            "attn.packed_lin": "attn.qkv_lin",
            "attn.ssmax_scale": "attn.sdpa.qassmax.scale",
            "attn.ssmax_gate": "attn.sdpa.qassmax.gate",
            "norm2": "mlp.0",
            "lin1": "mlp.1",
            "lin2": "mlp.3",
        }
        for old, new in replacements.items():
            if tail == old:
                return new
            if tail.startswith(f"{old}."):
                return f"{new}{tail[len(old) :]}"
        return tail

    if is_classifier:
        ignored_prefixes = (
            "row_embedding.y_reg_lin.",
            "icl_block.y_reg_lin.",
            "icl_block.reg_head.",
        )
        variant_replacements = (
            ("row_embedding.y_cls_lin.", "row_embedding.y_emb."),
            ("icl_block.y_cls_lin.", "icl_block.y_emb."),
            ("icl_block.cls_head.", "head.2."),
        )
    else:
        ignored_prefixes = (
            "row_embedding.y_cls_lin.",
            "icl_block.y_cls_lin.",
            "icl_block.cls_head.",
        )
        variant_replacements = (
            ("row_embedding.y_reg_lin.", "row_embedding.y_lin."),
            ("icl_block.y_reg_lin.", "icl_block.y_lin."),
            ("icl_block.reg_head.", "head.2."),
        )

    prefix_replacements = (
        *variant_replacements,
        ("gnn.post_lin.", "gnn.out_lin."),
        ("gnn.post_norm.", "gnn.out_norm."),
        ("icl_block.mlp.0.", "icl_block.norm."),
        ("icl_block.mlp.1.", "head.0."),
    )
    remapped: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if key == "q" or key.startswith(ignored_prefixes):
            continue

        if key.startswith("row_embedding.inducing_vectors."):
            layer = key.removeprefix("row_embedding.inducing_vectors.")
            key = f"row_embedding.col_layers.{layer}.inducing_points"
            value = value.squeeze(1)
        elif key == "row_embedding.readout_token":
            value = value.squeeze(0)
        else:
            for old_prefix, new_prefix, module in (
                (
                    "row_embedding.col_to_set_layers.",
                    "row_embedding.col_layers.",
                    "transformer_1.",
                ),
                (
                    "row_embedding.set_to_col_layers.",
                    "row_embedding.col_layers.",
                    "transformer_2.",
                ),
                (
                    "row_embedding.row_layers.",
                    "row_embedding.row_layers.",
                    "",
                ),
                ("icl_block.layers.", "icl_block.layers.", ""),
            ):
                if key.startswith(old_prefix):
                    layer, _, tail = key[len(old_prefix) :].partition(".")
                    key = (
                        f"{new_prefix}{layer}.{module}{_map_transformer(tail)}"
                    )
                    break

        for old_prefix, new_prefix in prefix_replacements:
            if key.startswith(old_prefix):
                key = f"{new_prefix}{key[len(old_prefix) :]}"
                break

        remapped[key] = value

    return remapped
