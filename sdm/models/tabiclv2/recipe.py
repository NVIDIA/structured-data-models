"""Default preprocessing recipes for the TabICLv2 model.

These factories compose shared :mod:`sdm.processing` processors into the
deterministic ``TableTensor``-to-model-input path of the original TabICLv2
regressor and classifier (``soda-inria/tabicl``). Steps that need processors
or stages not yet implemented are kept as commented placeholders and ``TODO``s.
"""

from sdm.processing import (
    MeanImpute,
    Recipe,
    SigmaClip,
    StandardScale,
)


def default_regression_recipe() -> Recipe:
    """Return the default single-estimator regression recipe.

    Reproduces the deterministic feature path of the original TabICLv2
    regressor: per-column mean imputation, standard scaling, and two-stage
    4-sigma outlier clipping. The target is standard-scaled and its inverse
    maps predictions back to the original space.
    """
    # TODO Add decode and to-numerical (categorical encoding).
    # TODO Add fixed clipping e.g. via lambda method to [-100, 100].
    # TODO Implement and enable the Identity, ConstantFilter (drops
    #   constant/unique columns, like TabICL UniqueFeatureFilter), and
    #   FeaturePermute processors.
    # TODO Wrap normalization in Choice([...]) and add n_estimators.
    # TODO Bundle classification via TaskDispatch on target and output.
    return Recipe(
        features=[
            MeanImpute(),
            # ConstantFilter(),
            StandardScale(epsilon=1e-6),
            SigmaClip(threshold=4.0),
            # FeaturePermute(method="latin"),
        ],
        target=[
            StandardScale(),
        ],
        output=[
            # Identity(),
        ],
    )


# Classification variant, commented until LabelShuffle and LabelDecode exist
# (SoftmaxTemperature and the feature steps already do). It mirrors the
# TabICLv2 classifier: the same feature transforms as regression, a per-member
# class shift on the target (its inverse un-shuffles the class logits), and
# softmax temperature plus label decode on the output.
#
# def default_classification_recipe() -> Recipe:
#     """Return the default single-estimator classification recipe."""
#     return Recipe(
#         features=[
#             MeanImpute(),
#             # ConstantFilter(),
#             StandardScale(epsilon=1e-6),
#             SigmaClip(threshold=4.0),
#             # FeaturePermute(method="latin"),
#         ],
#         target=[
#             # LabelShuffle(method="shift"),
#         ],
#         output=[
#             SoftmaxTemperature(temperature=0.9),
#             LabelDecode(),
#         ],
#     )
