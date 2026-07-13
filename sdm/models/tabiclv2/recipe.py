"""Default processing recipe for the TabICLv2 model."""

from sdm.processing import (
    CategoricalAlign,
    Choice,
    ClassShuffle,
    ConstantFilter,
    FeaturePermute,
    HardClip,
    Identity,
    MeanImpute,
    Power,
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

    Categorical feature vocabularies are fitted on context rows. Missing and
    unseen categories remain encoded as ``-1``. The target stays under
    semantic-type dispatch: classification targets are code-shuffled while
    regression targets are standardized and later inverse-transformed.
    """
    return Recipe(
        features=[
            StypeDispatch(
                numerical=Identity(),
                categorical=[
                    CategoricalAlign(order="sorted"),
                    ToNumerical(),
                ],
            ),
            MeanImpute(),
            ConstantFilter(),
            StandardScale(epsilon=1e-6),
            HardClip(min_value=-100.0, max_value=100.0),
            Choice(Identity(), Power()),
            SigmaClip(threshold=4.0),
            FeaturePermute(method="shift"),
        ],
        target=[
            StypeDispatch(
                numerical=StandardScale(),
                categorical=[
                    CategoricalAlign(order="sorted"),
                    ClassShuffle(method="shift"),
                ],
            ),
        ],
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
