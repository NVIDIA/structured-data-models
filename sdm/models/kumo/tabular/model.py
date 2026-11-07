# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ModuleDict

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm._inference import inference_mode
from sdm.cache import Cache
from sdm.models import ECOC, Callback, ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.models.kumo.tabular.row_embedding import RowEmbedding
from sdm.processing import AlignCategories

MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "small": {
        "cell_channels": 128,
        "num_embedding_layers": 4,
        "num_embedding_heads": 4,
        "num_inducing_points": 128,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 12,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": None,
    },
    "medium": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 24,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": 2,
    },
    "large": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 1024,
        "num_icl_layers": 24,
        "num_icl_heads": 16,
        "num_icl_key_value_heads_for_query": 2,
    },
}


class KumoTabular(ICLModel):
    r"""The tabular foundation model from `"NVIDIA Kumo Tabular Sets a New
    Accuracy-Efficiency Frontier for Tabular Prediction"
    <https://huggingface.co/blog/nvidia/kumo-tabular>`__.

    .. figure:: /images/kumo_tabular.svg
        :width: 100%

    :class:`KumoTabular` processes a table in two stages. First, an interleaved
    row/column encoder transforms raw table cells into fixed-size row
    representations. Numerical and categorical cells use separate learned
    Fourier frequencies and projections, together with a learned missingness
    projection. Each encoder stage consists of:

    * **Column-wise:** Each feature group is processed across rows using
      induced set attention. Both context and query cells attend only to
      context-row keys and values.
    * **Row-wise:** Each row's feature groups and learnable readout tokens
      attend to one another, combining feature interactions into a fixed-size
      row representation.

    Second, a dataset-wise in-context learning transformer processes the row
    representations and predicts each query target from the labeled context
    rows. The ``"large"`` model widens this transformer to 1024 channels and
    16 attention heads. Its four 256-channel readout tokens concatenate
    directly to that width without a projection layer.

    For regression tasks, :class:`KumoTabular` predicts 999 quantiles named
    ``"q001"`` through ``"q999"``.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized.
        size: The model size, one of ``"small"``, ``"medium"``, or
            ``"large"``. Defaults to ``"large"``.
        pretrained: Whether to load pretrained checkpoints.
        device: The device for model parameters. If ``None``, uses PyTorch's
            default device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_multi_target: ClassVar[bool] = False
    supports_context_only_columns: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None = None,
        size: Literal["small", "medium", "large"] = "large",
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)
        if Task.classification in self.tasks:
            self.ecoc = ECOC(max_classes=10)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _KumoTabular(
                num_classes=10 if task == Task.classification else 0,
                num_quantiles=999 if task == Task.regression else 0,
                device="meta" if pretrained else device,
                **MODEL_KWARGS[size],
            )

        if pretrained:
            self._load_from_pretrained(size, device=device)

        self._mark_dynamic = False

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def compile(self, *args: Any, **kwargs: Any) -> None:
        r"""Compile :meth:`predict` using :func:`torch.compile`.

        By default (``dynamic=False``), :meth:`predict` runs one compiled
        graph per device, precision and number of estimators for 2-D queries
        of any non-zero number of rows, also after refits on another number of
        context rows (at least two) that keep the fitted columns, their order
        (*e.g.*, the same ``generator`` seed), categories and classes. Query
        rows must be predicted independently of each other, and categorical
        features must depend only on category values, not on their codes, as
        in the default recipe. Query categories not seen in :meth:`fit` become
        missing. Empty queries, queries with callbacks or more than two
        dimensions, one-row contexts, and queries with categorical features
        until the next :meth:`fit` are predicted eagerly.

        Args:
            args: Positional arguments passed to :func:`torch.compile`.
            kwargs: Keyword arguments passed to :func:`torch.compile`, which
                default to ``dynamic=False`` and ``fullgraph=True``. Any other
                ``dynamic`` value compiles :meth:`predict` as is.
        """
        kwargs = {"dynamic": False, "fullgraph": True} | kwargs
        super().compile(*args, **kwargs)
        self._mark_dynamic = kwargs["dynamic"] is False

    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        y: Tensor | TableTensor,  # [..., R, 1]
        *args: Any,
        **kwargs: Any,
    ) -> None:
        r""":meta private:"""  # noqa: D415
        super().fit(x, y, *args, **kwargs)

        if (
            self._mark_dynamic
            and isinstance(x, TableTensor)
            and x.dim() == 2
            and x.categorical.size(-1) > 0
        ):
            with inference_mode("no_grad"):
                vocabulary = AlignCategories().fit(x)
            assert self._cache is not None
            self._cache = Cache(self._cache, vocabulary=vocabulary).freeze()

    def _load_from_pretrained(
        self,
        size: Literal["small", "medium", "large"],
        device: torch.device | str | None,
    ) -> KumoTabular:
        device = torch.get_default_device() if device is None else device

        for task, model in self.models.items():
            if task == Task.classification:
                filename = f"{size}/classifier.pt"
            else:
                assert task == Task.regression
                filename = f"{size}/regressor.pt"

            path = download_checkpoint(
                repo_id="nvidia/Kumo-Tabular",
                filename=filename,
                revision="v1.0.1",
                config_filename="config.json",
            )
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt, assign=True)

        return self

    def _predict_compiled(
        self,
        x: Tensor | TableTensor,  # [..., R, D]
        related_tables: RelatedTables | None,
        *,
        callbacks: Sequence[Callback] | None,
    ) -> TableTensor:  # Recipe-defined output shape.
        if not self._mark_dynamic:
            return super()._predict_compiled(
                x, related_tables, callbacks=callbacks
            )
        if x.dim() != 2 or related_tables is not None or callbacks:
            return self._predict(x, related_tables, callbacks=callbacks)

        assert self._compiled_predict is not None
        assert self._cache is not None
        if isinstance(x, TableTensor) and x.categorical.size(-1) > 0:
            # Fitted categories keep category sizes the same across queries;
            # fits before `compile` record none.
            vocabulary = self._cache.get("vocabulary")
            if vocabulary is None:
                return self._predict(x)
            x = cast(AlignCategories, vocabulary).transform(x)

        num_rows = x.size(-2)
        # Row attention would split larger chunks in the graph, which guards
        # on their size.
        chunk_size = min(
            self._network(cache).row_embedding.max_query_rows(
                cache=ECOC._model_cache(cache),
                device=x.device,
            )
            for cache in self._batch_caches()
        )
        # Empty queries have no row to repeat, and marks require at least two
        # rows per chunk and context.
        if num_rows == 0 or chunk_size < 2 or not self._mark_context_rows():
            return self._predict(x)

        outs: list[Tensor] = []
        for start in range(0, num_rows, chunk_size):
            end = min(start + chunk_size, num_rows)
            # Gather every chunk, so all chunks share one layout, and repeat
            # the row of single-row chunks: 1-row inputs would specialize.
            index = torch.arange(start, max(end, start + 2), device=x.device)
            chunk = x.index_select(-2, index.clamp_(max=end - 1))
            torch._dynamo.mark_dynamic(chunk, 0, min=2, max=chunk_size)
            if isinstance(chunk, TableTensor):
                # Marks reach inner tensors of the same rank only; the text
                # layout has one more dimension.
                layout = chunk.text._layout
                torch._dynamo.mark_dynamic(layout, 0, min=2, max=chunk_size)
            out = self._compiled_predict(chunk)
            outs.append(out[..., : end - start, :])
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-2)
        return cast(TableTensor, out)

    def _mark_context_rows(self) -> bool:
        # Marks do not survive copies, so mark before every call.
        for cache in self._batch_caches():
            icl_block = self._network(cache).icl_block
            for tensor, dim in icl_block.context_row_dims(
                ECOC._model_cache(cache)
            ):
                if tensor.size(dim) < 2:
                    return False  # Sizes 0 and 1 always specialize.
                torch._dynamo.mark_dynamic(tensor, dim, min=2, max=_MAX_ROWS)
        return True

    def _batch_caches(self) -> list[Cache]:
        assert self._cache is not None
        num_batches = cast(int, self._cache["num_batches"])
        return [cast(Cache, self._cache[i]) for i in range(num_batches)]

    def _network(self, cache: Cache) -> _KumoTabular:
        if cache["classes"] is None:
            return cast(_KumoTabular, self.models[Task.regression])
        return cast(_KumoTabular, self.models[Task.classification])

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        x_context_only: TableTensor | None,  # [..., R_context, D_only]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        *,
        categorical_mask: Tensor,
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

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            classes = cast("Tensor | None", cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        if classes is None:
            out = self.models[Task.regression](
                x=x,
                y=y,
                categorical_mask=categorical_mask,
                cache=cache,
            )
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out,
            )

        out = self.ecoc(
            model=self.models[Task.classification],
            x=x,
            y=y,
            num_classes=len(classes),
            num_members=x.size(0) if x.dim() > 2 else 1,
            cache=cache,
            generator=generator,
            categorical_mask=categorical_mask,
        )
        if cache is not None and cache.is_replaying:
            columns = cast(tuple[str, ...], cache["class_columns"])
        else:
            columns = tuple(str(i) for i in classes.tolist())

        if cache is not None and cache.is_recording:
            cache["class_columns"] = columns

        return TableTensor(
            columns={"numerical": columns},
            numerical=out,
        )


# PyTorch 2.14 requires an upper bound for dynamic sizes with a lower bound.
_MAX_ROWS = 2**31 - 1


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        cell_channels: int = 128,
        num_embedding_layers: int = 4,
        num_embedding_heads: int = 4,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 4,
        icl_channels: int = 512,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        num_icl_key_value_heads_for_query: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_frequencies=num_frequencies,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        if cell_channels * num_readout_tokens != icl_channels:
            self.row_project = Linear(
                cell_channels * num_readout_tokens,
                icl_channels,
                **factory_kwargs,
            )
        else:
            self.row_project = Identity()
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=icl_channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            num_key_value_heads_for_query=num_icl_key_value_heads_for_query,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        x = self.row_embedding(x, y, categorical_mask, cache=cache)
        x = self.row_project(x)
        return self.icl_block(x=x, y=y, cache=cache)

    def forward_train(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        num_columns: Tensor | None = None,  # [...]
        train_size: Tensor | None = None,  # [...], int32 lengths in padded y
    ) -> Tensor:  # [..., R or R_test, num_classes or num_quantiles]
        x = self.row_embedding(
            x=x,
            y=y,
            categorical_mask=categorical_mask,
            num_columns=num_columns,
            train_size=train_size,
        )
        x = self.row_project(x)
        return self.icl_block(x=x, y=y, train_size=train_size)
