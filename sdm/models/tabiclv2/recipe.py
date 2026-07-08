"""Default preprocessing recipe for the TabICLv2 model.

The factory composes shared :mod:`sdm.processing` processors into the
deterministic ``TableTensor``-to-model-input path of the original TabICLv2
model (``soda-inria/tabicl``).

The recipe serves regression and classification through a shared feature
pipeline and task-dependent output processing. Per-member normalization via
``Choice`` plus ``n_estimators`` remains planned.

Steps that need processors not implemented yet (``ConstantFilter``,
``FeaturePermute``, ``LabelShuffle``, ``Choice``) are kept as commented
placeholders.

"""

from sdm.processing import (
    Identity,
    MeanImpute,
    Recipe,
    SigmaClip,
    SoftmaxTemperature,
    StandardScale,
    StypeDispatch,
    TaskDispatch,
    ToNumerical,
)


def default_recipe() -> Recipe:
    """Return the default regression and classification recipe.

    Categorical features are converted to numerical values before mean
    imputation, standard scaling, and two-stage 4-sigma clipping. Targets
    remain in their original space so the same invertible target pipeline
    supports both tasks. Classification logits use temperature-scaled softmax;
    regression outputs remain unchanged. Commented lines mark processors not
    implemented yet.
    """
    return Recipe(
        features=[
            StypeDispatch(
                numerical=Identity(),
                categorical=ToNumerical(),
            ),
            MeanImpute(),
            # ConstantFilter(),
            StandardScale(epsilon=1e-6),
            # TabICL also clips z-scores to [-100, 100] here; left out for now
            # (likely subsumed by SigmaClip); revisit after benchmarking.
            # Clip(min_value=-100.0, max_value=100.0),
            SigmaClip(threshold=4.0),
            # FeaturePermute(method="latin"),
        ],
        target=[
            # TODO: Restore numerical scaling with invertible dispatch (#202).
            Identity(),
        ],
        output=[
            TaskDispatch(
                classification=SoftmaxTemperature(temperature=0.9),
                regression=Identity(),
            ),
        ],
    )
