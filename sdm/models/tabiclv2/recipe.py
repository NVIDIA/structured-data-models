from sdm.processing import (
    AlignCategories,
    Choice,
    Clip,
    ClipSigma,
    DropConstantColumns,
    Identity,
    ImputeMean,
    PowerTransform,
    Recipe,
    ReduceEstimators,
    ShuffleCategories,
    ShuffleColumns,
    Softmax,
    Standardize,
    StypeDispatch,
    TaskDispatch,
    ToNumerical,
)


def default_recipe() -> Recipe:  # noqa: D103
    return Recipe(
        features=[
            StypeDispatch(
                categorical=[
                    AlignCategories(sort_by="value"),
                    ToNumerical(),
                ],
            ),
            StypeDispatch(
                numerical=[
                    ImputeMean(),
                    DropConstantColumns(),
                    Standardize(epsilon=1e-6),
                    Clip(min_value=-100.0, max_value=100.0),
                    Choice(Identity(), PowerTransform()),
                    ClipSigma(threshold=4.0),
                    ShuffleColumns(method="shift"),
                ],
            ),
        ],
        target=[
            StypeDispatch(
                categorical=[
                    AlignCategories(),
                    ShuffleCategories(method="shift"),
                ],
                numerical=Standardize(),
            ),
        ],
        output=[
            ReduceEstimators(method="mean"),
            TaskDispatch(
                classification=Softmax(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
