from sdm.processing import (
    CategoricalAlign,
    Choice,
    ClassShuffle,
    ConstantFilter,
    FeaturePermute,
    Identity,
    MeanImpute,
    Quantile,
    Recipe,
    SigmaClip,
    SoftmaxTemperature,
    StandardScale,
    StypeDispatch,
    TaskDispatch,
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
            StypeDispatch(
                numerical=Identity(),
                categorical=[
                    CategoricalAlign(),
                    ToNumerical(),
                ],
            ),
            MeanImpute(),
            ConstantFilter(),
            StandardScale(epsilon=1e-6),
            Choice(Identity(), Quantile(output_distribution="normal")),
            # TabICL also clips z-scores to [-100, 100] here; left out for now
            # (likely subsumed by SigmaClip); revisit after benchmarking.
            # Clip(min_value=-100.0, max_value=100.0),
            SigmaClip(threshold=4.0),
            FeaturePermute(method="shift"),
        ],
        target=[
            StypeDispatch(
                categorical=ClassShuffle(method="shift"),
                numerical=StandardScale(),
            ),
        ],
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
