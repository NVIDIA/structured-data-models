# ruff: noqa: D205
from typing import ClassVar

import torch

from sdm import RelatedTables, Stype, TableTensor
from sdm.cache import Cache
from sdm.models import Model
from sdm.processing import Recipe


class KumoRFM(Model):
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
        {Stype.numerical}
    )
    #:
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    #:
    supports_multi_target: ClassVar[bool] = False
    #:
    supports_related_tables: ClassVar[bool] = True

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

    def _forward(
        self,
        x: TableTensor,  # [..., R, C_1]
        y: TableTensor,  # [..., R_train, C_2]
        related_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> TableTensor:  # [..., R_test, num_classes or 999]
        out = torch.empty(
            (*x.size()[:-2], x.size(-2) - y.size(-1), 10),
            device=x.device,
        )
        return TableTensor.from_tensor(out)

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        raise NotImplementedError
