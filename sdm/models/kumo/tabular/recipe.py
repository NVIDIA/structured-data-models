from typing import Literal

import sdm.processing as sp

NumericalMissing = Literal["nan", "impute", "mix"]
NumericTransform = Literal["round_robin", "identity", "power", "quantile"]


def slot_recipe(
    slots: list[tuple[NumericTransform, int | None]],
    numerical_missing: NumericalMissing = "nan",
) -> sp.Recipe:
    r"""Default recipe whose estimator ``i`` uses ``slots[i]``.

    Each slot is ``(numeric_transform, shuffle_categories_max)``; the
    recipe is meant for ``num_estimators == len(slots)``.
    """
    transforms = [_numeric_transform(t) for t, _ in slots]
    shuffles: list[sp.Processor] = [
        sp.ShuffleCategories(max_categories=m)
        if m is not None
        else sp.Identity()
        for _, m in slots
    ]
    return _recipe(
        numerical_missing,
        transform=sp.Choice(*transforms, method="round_robin"),
        shuffle=[sp.Choice(*shuffles, method="round_robin")],
    )


def _numeric_transform(numeric_transform: NumericTransform) -> sp.Processor:
    if numeric_transform == "identity":
        return sp.Identity()
    if numeric_transform == "power":
        return sp.PowerTransform()
    if numeric_transform == "quantile":
        return sp.QuantileTransform(n_quantiles=100)
    return sp.Choice(sp.Identity(), sp.PowerTransform(), method="round_robin")


def default_recipe(
    numerical_missing: NumericalMissing = "nan",
    numeric_transform: NumericTransform = "round_robin",
    shuffle_categories_max: int | None = None,
) -> sp.Recipe:
    r"""Default recipe.

    ``numerical_missing`` keeps NaN, mean-imputes, or alternates both across
    estimators (``mix``). ``numeric_transform`` alternates identity and power
    (``round_robin``), uses one of them, or applies a uniform quantile map
    (``quantile``). ``shuffle_categories_max`` permutes the codes of
    categorical columns with at most that many levels per estimator.
    """
    shuffle: list[sp.Processor] = []
    if shuffle_categories_max is not None:
        shuffle = [sp.ShuffleCategories(max_categories=shuffle_categories_max)]
    return _recipe(
        numerical_missing,
        transform=_numeric_transform(numeric_transform),
        shuffle=shuffle,
    )


def _recipe(
    numerical_missing: NumericalMissing,
    transform: sp.Processor,
    shuffle: list[sp.Processor],
) -> sp.Recipe:
    missing: list[sp.Processor] = []
    if numerical_missing == "impute":
        missing = [sp.ImputeMean()]
    elif numerical_missing == "mix":
        missing = [
            sp.Choice(sp.Identity(), sp.ImputeMean(), method="round_robin")
        ]
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
