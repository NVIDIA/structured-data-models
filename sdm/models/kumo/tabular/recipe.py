from typing import Literal

import sdm.processing as sp

NumericalMissing = Literal["nan", "impute", "mix"]
NumericTransform = Literal["round_robin", "identity", "power", "quantile"]


def default_recipe(
    numerical_missing: NumericalMissing = "nan",
    numeric_transform: NumericTransform = "round_robin",
    shuffle_categories_max: int | None = None,
    interactions: bool = False,
) -> sp.Recipe:
    r"""Default recipe.

    ``numerical_missing`` keeps NaN, mean-imputes, or alternates both across
    estimators (``mix``). ``numeric_transform`` alternates identity and power
    (``round_robin``), uses one of them, or alternates identity, power and a
    uniform quantile map (``quantile``). ``shuffle_categories_max`` permutes
    the codes of categorical columns with at most that many levels per
    estimator. ``interactions`` appends pairwise products and differences on
    narrow numerical blocks.
    """
    missing: list[sp.Processor] = []
    if numerical_missing == "impute":
        missing = [sp.ImputeMean()]
    elif numerical_missing == "mix":
        missing = [
            sp.Choice(sp.Identity(), sp.ImputeMean(), method="round_robin")
        ]
    transform: sp.Processor
    if numeric_transform == "identity":
        transform = sp.Identity()
    elif numeric_transform == "power":
        transform = sp.PowerTransform()
    elif numeric_transform == "quantile":
        transform = sp.Choice(
            sp.Identity(),
            sp.PowerTransform(),
            sp.QuantileTransform(n_quantiles=100),
            method="round_robin",
        )
    else:
        transform = sp.Choice(
            sp.Identity(),
            sp.PowerTransform(),
            method="round_robin",
        )
    shuffle: list[sp.Processor] = []
    if shuffle_categories_max is not None:
        shuffle = [sp.ShuffleCategories(max_categories=shuffle_categories_max)]
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value", min_frequency=2),
                    *shuffle,
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    *missing,
                    sp.DropConstantColumns(),
                    sp.Standardize(eps=1e-6),
                    *([sp.PairwiseInteractions()] if interactions else []),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    transform,
                    sp.ClipSigma(threshold=4.0),
                    sp.FlipSign(),
                    sp.ShuffleColumns(method="latin"),
                    sp.SelectColumns(500, method="first"),
                ],
            ),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(),
                ],
            ),
        ],
        output=[
            sp.TaskDispatch(regression=sp.SortQuantiles()),
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(classification=sp.Softmax(temperature=1.0)),
        ],
    )
