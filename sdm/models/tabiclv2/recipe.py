from sdm.processing import (
    AlignCategories,
    Choice,
    Clip,
    ClipBySigma,
    DispatchByStype,
    DispatchByTask,
    DropConstant,
    Identity,
    ImputeMean,
    PowerTransform,
    Recipe,
    ReduceEstimators,
    ShuffleCategories,
    ShuffleColumns,
    Softmax,
    Standardize,
    ToNumerical,
)


def default_recipe() -> Recipe:
    """Return the task-aware default recipe of the TabICLv2 model.

    Composes shared :mod:`sdm.processing` processors into the
    ``TableTensor``-to-model-input path of the original TabICLv2 model
    (``soda-inria/tabicl``). The target semantic type selects the target and
    output routes. For regression, the target inverse receives the complete
    numerical model-output head. Missing and unseen categorical feature
    values remain encoded as ``-1``.
    """
    return Recipe(
        features=[
            DispatchByStype(
                numerical=Identity(),
                categorical=[
                    AlignCategories(),
                    ToNumerical(),
                ],
            ),
            DispatchByStype(  # TODO Support `id` as passthrough.
                numerical=[
                    ImputeMean(),
                    DropConstant(),
                    Standardize(epsilon=1e-6),
                    Clip(min_value=-100.0, max_value=100.0),
                    Choice(Identity(), PowerTransform()),
                    ClipBySigma(threshold=4.0),
                    ShuffleColumns(method="shift"),
                ],
            ),
        ],
        target=[
            DispatchByStype(
                categorical=[
                    AlignCategories(),
                    ShuffleCategories(method="shift"),
                ],
                numerical=Standardize(),
            ),
        ],
        output=[
            ReduceEstimators(method="mean"),
            DispatchByTask(
                classification=Softmax(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
