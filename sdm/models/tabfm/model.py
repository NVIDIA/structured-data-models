# ruff: noqa: D205

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from torch import Tensor
from torch.nn import ModuleDict

import sdm.processing as sp
from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.models.base import ICLModel
from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.models.tabfm.ckpt import remap_ckpt
from sdm.models.tabfm.icl import ICLBlock
from sdm.models.tabfm.row_embedding import RowEmbedding
from sdm.tensor.table import TableSchema


class TabFM(ICLModel):
    r"""The tabular foundation model from `"Introducing TabFM: A Zero-shot
    Foundation Model for Tabular Data" <https://research.google/blog/
    introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data>`__.

    .. figure:: /images/tabfm_light.png
        :figclass: light-only
        :width: 100%

    .. figure:: /images/tabfm_dark.png
        :figclass: dark-only
        :width: 100%

    Architecturally, :class:`TabFM` can be viewed as a scaled-up
    :class:`TabICLv2`-style model with Fourier cell embeddings, separate
    numerical and categorical cell projections, two repeated column/row
    interaction stages, :class:`~torch.nn.RMSNorm`-based transformer blocks and
    :class:`~sdm.nn.SwiGLU` feed-forward blocks.

    .. note::
        :class:`TabFM` model weights are distributed under a
        `non-commercial license <https://huggingface.co/google/
        tabfm-1.0.0-pytorch/blob/
        77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/LICENSE>`__.
        Users are expected to download the
        `checkpoint <https://huggingface.co/google/tabfm-1.0.0-pytorch>`__
        manually and use it in accordance with its license.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized. Pass a single task to avoid initializing
            separate ~1.6B parameter models.
        checkpoint_path: The local checkpoint path.
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
        task: TaskLike | None,
        checkpoint_path: str | Path | None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _TabFM(
                num_classes=10 if task == Task.classification else 0,
                device="meta" if checkpoint_path is not None else device,
            )

        if checkpoint_path is not None:
            self._load_from_pretrained(checkpoint_path, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return Recipe(
            features=[
                sp.StypeDispatch(
                    categorical=[
                        sp.AlignCategories(min_frequency=2),
                        sp.ToNumerical(),
                    ],
                ),
                sp.StypeDispatch(
                    numerical=[
                        sp.ImputeMean(),
                        sp.DropConstantColumns(),
                        sp.Standardize(epsilon=1e-6),
                        sp.Clip(min_value=-100.0, max_value=100.0),
                        sp.Choice(
                            sp.Identity(),
                            sp.PowerTransform(),
                            method="round_robin",
                        ),
                        sp.ClipSigma(threshold=4.0),
                        sp.ShuffleColumns(method="random"),
                        sp.SelectColumns(
                            max_columns=500,
                            method="round_robin",
                        ),
                    ],
                ),
            ],
            target=sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=sp.Standardize(),
            ),
            output=[
                sp.ReduceEstimators(method="mean"),
                sp.TaskDispatch(
                    classification=sp.Softmax(temperature=0.9),
                ),
            ],
        )

    def _load_from_pretrained(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str | None,
    ) -> TabFM:
        from safetensors.torch import load_file  # noqa: PLC0415

        device = torch.get_default_device() if device is None else device

        assert len(self.models) == 1
        for task, model in self.models.items():
            ckpt = remap_ckpt(
                ckpt=load_file(checkpoint_path, device=str(device)),
                is_classifier=task == Task.classification,
            )
            model.load_state_dict(ckpt, strict=True, assign=True)

        return self

    def forward(self, *args: Any, **kwargs: Any) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        x_context = kwargs["x_context"] if "x_context" in kwargs else args[0]
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        kwargs["_schema"] = x_context.schema
        return super().forward(*args, **kwargs)

    def fit(self, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        x = kwargs["x"] if "x" in kwargs else args[0]
        if not isinstance(x, TableTensor):
            x = TableTensor.from_tensor(x)
        kwargs["_schema"] = x.schema
        return super().fit(*args, **kwargs)

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
    ) -> TableTensor:  # [..., R_query, num_classes or 1]

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
            classes = cast(Tensor | None, cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if classes is not None and len(classes) > 10:
            raise ValueError(
                f"{self.__class__.__name__!r} only supports up to 10 classes "
                f"(got {len(classes)})"
            )

        if cache is None or cache.is_recording:
            assert x_context is not None
            schema: TableSchema = kwargs["_schema"]
            categorical_columns = set(schema.columns[Stype.categorical])
            categorical_mask = torch.tensor(
                [
                    column in categorical_columns
                    for column in x_context.columns[Stype.numerical]
                ],
                device=x.device,
                dtype=torch.bool,
            )
            if cache is not None:
                cache["categorical_mask"] = categorical_mask
        else:
            categorical_mask = cast(Tensor, cache["categorical_mask"])
        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        task = Task.classification if classes is not None else Task.regression
        out = self.models[task](x, y, categorical_mask, cache=cache)

        if classes is None:
            return TableTensor(
                columns={Stype.numerical: ["pred"]},
                numerical=out,
            )

        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out[..., : len(classes)],
        )


class _TabFM(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int = 256,
        num_embedding_layers: int = 3,
        num_embedding_repeats: int = 2,
        num_embedding_col_heads: int = 4,
        num_embedding_row_heads: int = 8,
        num_inducing_points: int = 256,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 8,
        num_icl_layers: int = 24,
        num_icl_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.cell_embedding = CellEmbedding(
            channels=channels,
            group_size=group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )
        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_embedding_layers,
            num_repeats=num_embedding_repeats,
            num_col_heads=num_embedding_col_heads,
            num_row_heads=num_embedding_row_heads,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=max(1, num_classes),
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, 1 or num_classes]
        x = self.cell_embedding(x, categorical_mask)
        x = self.row_embedding(x, y, cache=cache)
        return self.icl_block(x, y, cache=cache)
