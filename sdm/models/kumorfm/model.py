# ruff: noqa: D205
from typing import ClassVar

import torch

from sdm import RelatedTables, TableTensor
from sdm.cache import Cache
from sdm.models import Model
from sdm.processing import (
    CategoryShuffle,
    ClassDecode,
    EstimatorMean,
    Identity,
    QuantileDecode,
    Recipe,
    SoftmaxTemperature,
    StandardScale,
    StypeDispatch,
    TargetDecode,
    TaskDispatch,
)


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
    supports_related_tables: ClassVar[bool] = True

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables | None,
        related_query_tables: RelatedTables | None,
        cache: Cache | None,
    ) -> TableTensor:  # [..., R_query, *]
        raise NotImplementedError

    @classmethod
    def default_recipe(cls) -> Recipe:
        """Return the minimal task-aware KumoRFM output Recipe.

        Regression follows KumoRFM v2.1's median path: sort and select the
        model quantile head, invert each estimator's fitted target transform,
        and then average estimators. Binary classification restores shuffled
        class columns before averaging logits and applying softmax.
        """
        return Recipe(
            target=[
                StypeDispatch(
                    categorical=CategoryShuffle(method="random"),
                    numerical=StandardScale(),
                ),
            ],
            output=[
                TaskDispatch(
                    classification=ClassDecode(),
                    regression=[
                        QuantileDecode(method="median"),
                        TargetDecode(),
                    ],
                ),
                EstimatorMean(),
                TaskDispatch(
                    classification=SoftmaxTemperature(),
                    regression=Identity(),
                ),
            ],
        )
