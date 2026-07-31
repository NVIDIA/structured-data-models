# ruff: noqa: D205
from collections.abc import Sequence
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import RelatedTables, Relationship, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.recipe import default_recipe
from sdm.models.kumorfm.task import TaskGraph
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.processing import Recipe


def peak():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_host_memory_stats()
    torch.cuda.synchronize(device)
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    print(f"Allocated: {peak_alloc / 1000**2:.1f}MB")
    print(f"Reserved: {peak_reserved / 1000**2:.1f}MB")


class KumoRFM(ICLModel):
    r"""An adapted and simplified version of the relational foundation model
    from the `"KumoRFM-2: Scaling Foundation Models for Relational Learning"
    <https://arxiv.org/abs/2604.12596>`_ paper.

    .. figure:: /images/rfm_light.svg
        :figclass: light-only
        :width: 100%

    .. figure:: /images/rfm_dark.svg
        :figclass: dark-only
        :width: 100%

    :class:`KumoRFM` extends the in-context learning structure of tabular
    foundation models from single tables to relational, multi-table inputs.
    It processes task rows together with one or more related tables, avoiding
    manual flattening of relational data into a single table.

    This implementation follows the high-level KumoRFM-2 architecture.
    It consists of three stages:

    * **Intra-table row embeddings:** Each related table is embedded
      independently with a :class:`TabICLv2`-style row embedding stack.
      Target information is injected by distributing context targets over the
      relational graph.
    * **Inter-table message passing:** A schema-agnostic GNN exchanges the row
      embeddings along pre-defined relationships for ``num_hops`` rounds,
      before it reads out the rows linked to the task table.
    * **In-context learning over samples:** The readout embeddings for context
      and query task rows are processed by a :class:`TabICLv2`-style
      dataset-level ICL block.
      Context rows carry target information, while query rows attend to the
      labeled context to produce class logits or regression quantiles.

    .. testcode::

        from sdm import RelatedTables, TableTensor
        from sdm.models import KumoRFM

        task_table = TableTensor.from_columns(
            {"user_id": [0, 1, 2, 3], "churn": [True, False, True, False]},
            stypes={"user_id": "id", "churn": "categorical"},
            device="cuda",
        )

        related_tables = RelatedTables(
            tables={
                "users": TableTensor.from_columns(
                    {"user_id": [0, 1, 2, 3], "age": [42, 23, 31, 26]},
                    stypes={"user_id": "id", "age": "numerical"},
                    device="cuda",
                ),
                "orders": TableTensor.from_columns(
                    {
                        "user_id": [0, 0, 1, 3, 3, 3],
                        "amount": [9.99, 4.99, 12.99, 7.99, 3.99, 5.99],
                    },
                    stypes={"user_id": "id", "amount": "numerical"},
                    device="cuda",
                ),
            },
            relationships=[{
                "left_table": "orders",
                "left_columns": "user_id",
                "right_table": "users",
                "right_columns": "user_id",
            }],
            task_links=[{
                "task_columns": "user_id",
                "table": "users",
                "table_columns": "user_id",
            }],
        )

        x_context = task_table[:2].drop_columns("churn")
        y_context = task_table[:2, "churn"]
        x_query = task_table[2:].drop_columns("churn")

        related_context_tables = related_tables.replace_tables({
            "users": related_tables.tables["users"][:2],
            "orders": related_tables.tables["orders"][:3],
        })
        related_query_tables = related_tables.replace_tables({
            "users": related_tables.tables["users"][2:],
            "orders": related_tables.tables["orders"][3:],
        })

        model = KumoRFM(device="cuda")

        # Default in-context learning forward pass:
        out = model(
            x_context=x_context,
            y_context=y_context,
            x_query=x_query,
            related_context_tables=related_context_tables,
            related_query_tables=related_query_tables,
            num_hops=1,
        )

        # Fit+Predict forward pass via key/value caching:
        model.fit(x_context, y_context, related_context_tables, num_hops=1)
        out = model.predict(x_query, related_query_tables)

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.datetime}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
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
                repo_id="nvidia/kumorfm",
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

        print("After Preprocessing")
        peak()

        num_classes: int | None = None  # Extract `y` as tensor:
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
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

        context: TaskGraph | None = None
        if x_context is not None:
            if related_context_tables is None:
                raise ValueError(
                    f"{self.__class__.__name__!r} requires related tables"
                )
            context = TaskGraph.from_input(
                x=x_context,
                related_tables=related_context_tables,
                num_hops=num_hops,
            )
            if cache is not None and cache.is_recording:
                cache["relationships"] = context.related_tables.relationships
                cache["num_hops"] = context.num_hops

        query: TaskGraph | None = None
        if x_query is not None:
            if related_query_tables is None:
                raise ValueError(
                    f"{self.__class__.__name__!r} requires related tables"
                )
            if context is not None:
                relationships = context.related_tables.relationships
                num_hops = context.num_hops
            else:
                assert cache is not None
                assert cache.is_replaying
                relationships = cast(
                    Sequence[Relationship], cache["relationships"]
                )
                num_hops = cast(int, cache["num_hops"])
            query = TaskGraph.from_input(
                x=x_query,
                related_tables=RelatedTables(
                    tables=related_query_tables.tables,
                    relationships=relationships,
                    task_links=related_query_tables.task_links,
                ),
                num_hops=num_hops,
            )

        print("After task graph creation")
        peak()

        # TODO Support computing relative time.
        # TODO Inject random heterogeneous GNN.

        # Reason within each Table ############################################
        if context is not None:
            table_names = list(context.related_tables.tables)
            readout_table = context.readout_table
        else:
            assert query is not None
            table_names = list(query.related_tables.tables)
            readout_table = query.readout_table

        xs_context: dict[str, Tensor] = {}
        xs_query: dict[str, Tensor] = {}
        for name in table_names:
            x_context_i = task_row_i = None
            if context is not None:
                x_context_i = context.related_tables.tables[name].numerical
                task_row_i = context.task_row_by_table[name]
                if name == readout_table:  # Inject task features:
                    assert x_context is not None
                    x_context_i = self._inject_task(
                        x=x_context_i,
                        task=x_context.numerical,
                        readout_index=context.readout_index,
                    )

            x_query_i = None
            if query is not None and name in query.related_tables.tables:
                x_query_i = query.related_tables.tables[name].numerical
                if name == readout_table:  # Inject task features:
                    assert x_query is not None
                    x_query_i = self._inject_task(
                        x=x_query_i,
                        task=x_query.numerical,
                        readout_index=query.readout_index,
                    )

            print(f"Before embed {name}")
            peak()

            xs_context[name], xs_query[name] = self._embed_table(
                x_context=x_context_i,
                x_query=x_query_i,
                y=y,
                task_row=task_row_i,
                num_classes=num_classes,
                cache_key=f"table_{name}",
                cache=cache,
                generator=generator,
            )

            print(f"After embed {name}")
            peak()

        # Inter-Message Passing ###############################################
        gnn_cache = cache or Cache()
        if context is not None:
            x_context: Tensor = torch.cat(
                [xs_context[name] for name in context.related_tables.tables],
                dim=-2,
            )
            del xs_context
            x_context = self.gnn(
                x=x_context,
                graph=context.graph,
                readout_table=context.readout_table,
                readout_index=context.readout_index,
                num_hops=context.num_hops,
                cache=gnn_cache,
                generator=generator,
            )

        if query is not None:
            x_query: Tensor = torch.cat(
                [xs_query[name] for name in query.related_tables.tables],
                dim=-2,
            )
            del xs_query
            x_query = self.gnn(
                x=x_query,
                graph=query.graph,
                readout_table=query.readout_table,
                readout_index=query.readout_index,
                num_hops=query.num_hops,
                cache=gnn_cache.freeze() if cache is None else cache,
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
        return self.icl_block(
            x=x,
            y=y,
            num_classes=num_classes,
            cache=cache,
        )

    def _embed_table(
        self,
        x_context: Tensor | None,
        x_query: Tensor | None,
        y: Tensor,
        task_row: Tensor | None,
        num_classes: int | None,
        cache_key: str,
        cache: Cache | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        # Embed context and query rows jointly per table. Targets are injected
        # by distributing them to related tables via task-row assignment:
        _cache: Cache | None = None
        if cache is not None and cache.is_recording:
            _cache = Cache()
        elif cache is not None:
            _cache = cast(Cache, cache[cache_key])

        train_mask: Tensor | None = None
        if x_context is not None:
            assert task_row is not None
            x = x_context
            train_mask = task_row >= 0
            y = y[task_row[train_mask]]
            if x_query is not None:
                x = torch.cat([x, x_query], dim=-2)
                test_mask = train_mask.new_zeros(x_query.size(-2))
                train_mask = torch.cat([train_mask, test_mask])
        else:
            assert x_query is not None
            x = x_query

        print("Before row_embedding", x.size(), y.size())
        peak()

        x = self.row_embedding(
            x=x,
            y=y,
            train_mask=train_mask,
            max_keys=self.max_train_size,
            num_classes=num_classes,
            cache=_cache,
            generator=generator,
        )

        print("After row_embedding", x.size())
        peak()

        if cache is not None and cache.is_recording:
            cache[cache_key] = cast(Cache, _cache)

        sections = [
            x_context.size(-2) if x_context is not None else 0,
            x_query.size(-2) if x_query is not None else 0,
        ]
        return x.split(sections, dim=-2)

    def _inject_task(
        self,
        x: Tensor,
        task: Tensor,
        readout_index: Tensor,
    ) -> Tensor:

        if task.size(-1) == 0:
            return x

        x = torch.cat(
            [x, x.new_zeros(*x.size()[:-1], task.size(-1))],
            dim=-1,
        )
        x[..., readout_index, -task.size(-1) :] = task
        return x


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
            ("icl_block.cls_head.", "icl_block.head.2."),
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
            ("icl_block.reg_head.", "icl_block.head.2."),
        )

    prefix_replacements = (
        *variant_replacements,
        ("gnn.post_lin.", "gnn.out_lin."),
        ("gnn.post_norm.", "gnn.out_norm."),
        ("icl_block.mlp.0.", "icl_block.norm."),
        ("icl_block.mlp.1.", "icl_block.head.0."),
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
