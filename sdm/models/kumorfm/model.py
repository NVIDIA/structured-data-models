# ruff: noqa: D205

from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models import Model
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.models.kumorfm.table_hop_encoder import TableHopEncoder
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.processing import Recipe


class KumoRFM(Model):
    r"""The adapted relational foundation model from the `"KumoRFM-2: Scaling
    Foundation Models for Relational Learning"
    <https://arxiv.org/abs/2604.12596>`_ paper.

    .. image:: https://arxiv.org/html/2604.12596v1/x3.png
        :align: center
        :width: 100%

    Args:
        pretrained: Whether to load the pretrained checkpoint. Checkpoint
            loading is not yet available.
        device: The device.
    """

    #:
    supports_related_tables: ClassVar[bool] = True

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if pretrained:
            raise NotImplementedError(
                "Pretrained KumoRFM checkpoints are not available yet"
            )

        self.cls_model = _KumoRFM(
            num_classes=10,
            num_quantiles=0,
            device=device,
        )
        self.reg_model = _KumoRFM(
            num_classes=0,
            num_quantiles=999,
            device=device,
        )

        self.eval()

    def forward(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
    ) -> Tensor:
        r"""Run a single KumoRFM estimator."""
        if num_estimators != 1:
            raise ValueError("KumoRFM currently supports one estimator")
        return super().forward(
            x=x,
            y=y,
            related_tables=related_tables,
            recipe=recipe,
            num_estimators=num_estimators,
        )

    def _forward(
        self,
        x: Tensor,  # [R, C]
        y: Tensor,  # [R_train]
        related_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> Tensor:  # [R - R_train, num_classes or num_quantiles]
        r"""Run entity prediction over sampled related tables."""
        if y.is_complex():
            raise TypeError("KumoRFM targets cannot be complex")

        model = self.reg_model if y.is_floating_point() else self.cls_model
        if torch.is_inference_mode_enabled():
            with torch.inference_mode(False), torch.no_grad():
                out = model(x, y, related_tables, cache=cache)
            return out.clone()
        return model(x, y, related_tables, cache=cache)

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        raise NotImplementedError

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        device_repr = f"device={device}" if device.type != "cpu" else ""
        return f"{self.__class__.__name__}({device_repr})"


class _KumoRFM(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        cell_channels: int = 128,
        channels: int = 512,
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
        if (num_classes > 0) == (num_quantiles > 0):
            raise ValueError(
                "Exactly one of `num_classes` and `num_quantiles` must be "
                "positive"
            )
        if channels != cell_channels * num_readout_tokens:
            raise ValueError(
                "`channels` must equal `cell_channels * num_readout_tokens`"
            )

        self.num_classes = num_classes
        self.num_quantiles = num_quantiles

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.table_hop_encoder = TableHopEncoder(row_embedding)
        self.gnn = InvariantGNN(channels=channels, **factory_kwargs)
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(
                2 * channels,
                num_classes or num_quantiles,
                **factory_kwargs,
            ),
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        related_tables: RelatedTables | None,
        *,
        cache: Cache | None = None,
    ) -> Tensor:
        if cache is not None:
            raise NotImplementedError(
                "KumoRFM fit/predict caching is not available yet"
            )
        if related_tables is None:
            raise ValueError("KumoRFM requires related tables")
        if len(related_tables.task_links) != 1:
            raise ValueError(
                "KumoRFM entity prediction requires exactly one task link"
            )
        if y.numel() == 0:
            raise ValueError("KumoRFM requires at least one context target")

        parameter = self.gnn.src_lin.weight
        x = x.to(device=parameter.device, dtype=parameter.dtype)
        if self.num_classes:
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if y.dtype not in integer_dtypes:
                raise TypeError("Classification targets must be integers")
            y = y.to(device=parameter.device, dtype=torch.long)
            if bool((y < 0).any()) or bool((y >= self.num_classes).any()):
                raise ValueError(
                    "Classification targets must be between 0 and "
                    f"{self.num_classes - 1}"
                )
        else:
            if not y.is_floating_point():
                raise TypeError("Regression targets must be floating point")
            y = y.to(device=parameter.device, dtype=parameter.dtype)
        generator = torch.Generator(device=parameter.device).manual_seed(42)

        encoded = self.table_hop_encoder(
            x=x,
            y=y,
            related_tables=related_tables,
            max_keys=20_000,
            generator=generator,
        )
        entity_table = related_tables.task_links[0].table
        x = self.gnn(
            x_dict=encoded.x_dict,
            edge_index_dict=encoded.edge_index_dict,
            readout_table=entity_table,
            num_hops=encoded.num_hops,
            generator=generator,
        )
        x = x.index_select(0, encoded.root_index)
        x = self.icl_block(x=x, y=y)
        return self.head(x)
