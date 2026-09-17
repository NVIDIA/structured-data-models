# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor

# Scalar float32 statistics from standardizer.pkl at the checkpoint revision
# abff20a58834638b28227ff4ab934f26206e4b09 in nvidia/nv-tesseract-forecasting.
# Keeping the values here avoids executing a pickle or requiring joblib.
_MEAN = -5.671202659606934
_SCALE = 8.693312644958496


class _Standardize(Processor, InvertibleMixin):
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
    No statistics are fitted on future rows. Reversible instance normalization
    remains inside the model, as it is part of the checkpoint architecture.
    """
    return sp.Recipe(
        features=_Standardize(),
        target=_Standardize(),
        output=sp.ReduceEstimators(method="mean"),
    )
