# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor

# Scalar float32 statistics from standardizer.pkl at the checkpoint revision
# abff20a58834638b28227ff4ab934f26206e4b09 in nvidia/Kumo-Forecast.
# Keeping the values here avoids executing a pickle or requiring joblib.
_MEAN = -5.671202659606934
_SCALE = 8.693312644958496


class _Standardize(Processor, InvertibleMixin):
    """Apply the released checkpoint's frozen training-set statistics.

    These are checkpoint metadata, not statistics of the inference context.
    Fitting a new standardizer would change the SDK's preprocessing contract.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(
            numerical=(table.numerical - _MEAN) / _SCALE
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical * _SCALE + _MEAN)


def default_recipe() -> sp.Recipe:
    """Preserve chronology and apply the released fixed normalization.

    The SDK's ``1e-8`` scale epsilon rounds away in its float32 standardizer.
    The fixed statistics come from the training artifact ``standardizer.pkl``
    distributed with the released weights. They are deliberately not refitted
    on each inference context. Data-dependent, per-channel normalization is
    already performed by RevIN inside the model, using observed context only.

    Although both operations are invertible, removing the fixed transform is
    not equivalent: RevIN's additive epsilon is applied in standardized units,
    which affects low-variance inputs. Keeping both preserves SDK inference.
    Checkpoints fine-tuned with different preprocessing need a matching recipe.
    """
    return sp.Recipe(
        features=_Standardize(),
        target=_Standardize(),
        output=sp.ReduceEstimators(method="mean"),
    )
