from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter

from sdm.models.tabfm.cell_embedding import CellEmbedding
from sdm.models.tabfm.column import _ColumnEmbedding
from sdm.models.tabfm.icl import ICLearning
from sdm.models.tabfm.row import _RowInteraction
from sdm.models.tabfm.target_embedding import TargetEmbedding


class _TabFM(torch.nn.Module):
    """Assemble the checkpoint-compatible TabFM v1.0.0 neural core."""

    def __init__(
        self,
        *,
        embed_dim: int = 8,
        max_classes: int = 10,
        col_num_blocks: int = 2,
        col_num_heads: int = 2,
        col_num_inducing_points: int = 4,
        row_num_blocks: int = 2,
        row_num_heads: int = 2,
        row_num_cls: int = 2,
        icl_num_blocks: int = 2,
        icl_num_heads: int = 2,
        feedforward_factor: int = 2,
        feature_group_size: int = 3,
        num_frequencies: int = 32,
        decoder_hidden_channels: int | None = None,
        is_classifier: bool = True,
        row_chunk_size: int | None = 4096,
        col_chunk_size: int | None = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        feedforward_channels = embed_dim * feedforward_factor
        icl_channels = embed_dim * row_num_cls
        if decoder_hidden_channels is None:
            decoder_hidden_channels = 2 * icl_channels

        self.max_classes = max_classes
        self.is_classifier = is_classifier
        self.cell_chunk_size = row_chunk_size
        self.cell_embedder = CellEmbedding(
            channels=embed_dim,
            group_size=feature_group_size,
            num_frequencies=num_frequencies,
            **factory_kwargs,
        )
        self.target_embedder = TargetEmbedding(
            channels=embed_dim,
            is_classifier=is_classifier,
            max_classes=max_classes,
            **factory_kwargs,
        )
        self.col_embedder = _ColumnEmbedding(
            channels=embed_dim,
            num_blocks=col_num_blocks,
            num_heads=col_num_heads,
            feedforward_channels=feedforward_channels,
            num_inducing_points=col_num_inducing_points,
            col_chunk_size=col_chunk_size,
            **factory_kwargs,
        )
        self.col_embedder_2 = _ColumnEmbedding(
            channels=embed_dim,
            num_blocks=col_num_blocks,
            num_heads=col_num_heads,
            feedforward_channels=feedforward_channels,
            num_inducing_points=col_num_inducing_points,
            col_chunk_size=col_chunk_size,
            **factory_kwargs,
        )
        self.row_interactor = _RowInteraction(
            num_blocks=row_num_blocks,
            channels=embed_dim,
            num_heads=row_num_heads,
            feedforward_channels=feedforward_channels,
            num_cls=row_num_cls,
            output_full=True,
            row_chunk_size=row_chunk_size,
            **factory_kwargs,
        )
        self.row_interactor_2 = _RowInteraction(
            num_blocks=row_num_blocks,
            channels=embed_dim,
            num_heads=row_num_heads,
            feedforward_channels=feedforward_channels,
            num_cls=row_num_cls,
            output_full=False,
            row_chunk_size=row_chunk_size,
            **factory_kwargs,
        )
        self.cls_tokens = Parameter(
            torch.zeros(row_num_cls, embed_dim, **factory_kwargs)
        )
        self.icl_predictor = ICLearning(
            channels=icl_channels,
            num_blocks=icl_num_blocks,
            num_heads=icl_num_heads,
            feedforward_channels=icl_channels * feedforward_factor,
            decoder_hidden_channels=decoder_hidden_channels,
            is_classifier=is_classifier,
            max_classes=max_classes,
            **factory_kwargs,
        )

    def forward(
        self,
        features: Tensor,
        targets: Tensor,
        context_size: Tensor,
        categorical_mask: Tensor | None = None,
        active_features: Tensor | None = None,
    ) -> Tensor:
        """Return predictions for every context and query row."""
        if features.is_complex():
            raise ValueError("features must not be complex")
        features = features.nan_to_num(nan=-100.0).to(
            dtype=self.cls_tokens.dtype
        )
        batch_size, num_rows, _ = features.shape
        row_index = torch.arange(num_rows, device=features.device)
        context = row_index[None, :] < context_size[:, None]
        targets = targets.where(context.expand_as(targets), 0)

        if categorical_mask is None:
            categorical_mask = torch.zeros(
                batch_size,
                features.size(-1),
                dtype=torch.bool,
                device=features.device,
            )
        if features.size(-1) == 0:
            embedding = features.new_empty(
                batch_size,
                num_rows,
                0,
                self.cls_tokens.size(-1),
            )
        else:
            if (
                self.cell_chunk_size is None
                or num_rows <= self.cell_chunk_size
            ):
                embedding = self.cell_embedder(
                    features,
                    categorical_mask,
                    active_features,
                )
            else:
                embedding = torch.cat(
                    [
                        self.cell_embedder(
                            chunk,
                            categorical_mask,
                            active_features,
                        )
                        for chunk in features.split(
                            self.cell_chunk_size, dim=1
                        )
                    ],
                    dim=1,
                )
        embedding = embedding + self.target_embedder(
            targets,
            context_size,
        ).unsqueeze(-2)
        embedding = self.col_embedder(embedding, context_size)

        cls_tokens = self.cls_tokens.expand(batch_size, num_rows, -1, -1)
        embedding = torch.cat([cls_tokens, embedding], dim=2)
        embedding = self.row_interactor(embedding, active_features)
        embedding = self.col_embedder_2(embedding, context_size)
        representations = self.row_interactor_2(embedding, active_features)
        return self.icl_predictor(representations, targets, context_size)
