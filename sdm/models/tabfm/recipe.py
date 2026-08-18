from collections.abc import Collection

import torch
from torch import Tensor

import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.models.tabfm._scheduling import _ShiftClasses, _ShuffleFeatures


def _categorical_mask(
    table: TableTensor,
    categorical_columns: Collection[str],
) -> Tensor:
    """Align pre-recipe categorical column names to transformed columns."""
    categorical_columns = frozenset(categorical_columns)
    return torch.tensor(
        [
            column in categorical_columns
            for column in table.columns[Stype.numerical]
        ],
        dtype=torch.bool,
        device=table.device,
    )


def default_recipe() -> sp.Recipe:
    """Return the basic Google-compatible TabFM recipe.

    Numerical and categorical features are supported. Multi-estimator member
    schedules require one vectorized recipe bind; schedules are local to each
    bind and are not equivalent across sequential one-member binds. Datetime
    expansion, categorical-value permutations, feature crosses/SVD, row
    subsampling, and fitted OOF/NNLS/calibration ensembles are excluded. Model
    caching is outside this preprocessing recipe. The literal string ``"nan"``
    is an ordinary category, not a missing value marker.
    """
    return sp.Recipe(
        features=[
            # The categorical route is emitted before passthrough numerical
            # columns, matching Google's categorical-first conversion.
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(
                        sort_by="appearance",
                        min_frequency=2,
                    ),
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    sp.ImputeMean(),
                    sp.Standardize(epsilon=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(
                        sp.Identity(),
                        sp.PowerTransform(),
                        method="round_robin",
                    ),
                    sp.ClipSigma(threshold=4.0),
                    _ShuffleFeatures(),
                ],
                id=sp.Identity(),
            ),
        ],
        target=sp.StypeDispatch(
            categorical=[
                sp.AlignCategories(sort_by="value"),
                _ShiftClasses(),
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
