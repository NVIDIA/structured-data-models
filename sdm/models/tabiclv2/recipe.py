"""Default preprocessing recipe for the TabICLv2 model.

The factory composes shared :mod:`sdm.processing` processors into the
``TableTensor``-to-model-input path of the original TabICLv2 model
(``soda-inria/tabicl``).
"""

from sdm.processing import (
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
    """Return the default regression and classification recipe."""
    return Recipe(
        features=[
            StypeDispatch(
                numerical=Identity(),
                categorical=[CategoricalAlign(), CategoricalImpute(), ToNumerical()],
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
                numerical=StandardScale(),
                categorical=ClassShuffle(method="shift"),
            ),
        ],
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
