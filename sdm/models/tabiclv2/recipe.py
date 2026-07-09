"""Default preprocessing recipe for the TabICLv2 model.

The factory composes shared :mod:`sdm.processing` processors into the
``TableTensor``-to-model-input path of the original TabICLv2 model
(``soda-inria/tabicl``).

The recipe serves regression and classification through a shared feature
pipeline, a target pipeline dispatched by the target semantic type, and
task-dependent output processing. Stochastic steps draw from the global
CPU generator at fit time; seed with :func:`torch.manual_seed` to make a
fitted recipe reproducible.
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
    """Return the default regression and classification recipe.

    Categorical features are converted to numerical values before mean
    imputation, constant-feature filtering, standard scaling, a drawn
    per-member normalization, two-stage 4-sigma outlier clipping, and a
    drawn cyclic feature shift. Numerical targets are standard-scaled and
    categorical targets are class-shuffled. Classification logits use
    temperature-scaled softmax; regression outputs remain unchanged.
    """
    return Recipe(
        features=[
            StypeDispatch(
                numerical=Identity(),
                categorical=ToNumerical(),
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
