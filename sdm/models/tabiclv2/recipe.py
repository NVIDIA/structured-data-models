"""Default preprocessing recipes for the TabICLv2 model.

Each factory composes shared :mod:`sdm.processing` processors into the
deterministic ``TableTensor``-to-model-input path of the original TabICLv2
model (``soda-inria/tabicl``).

Supported now:

- :func:`default_regression_recipe` -- single-estimator regression.

Planned, staged towards one task-aware recipe:

- per-member normalization via ``Choice`` plus ``n_estimators``.
- a single recipe that serves both regression and classification,
  dispatching the target and output roles per task with ``TaskDispatch``.
  The end-state is sketched (commented) at the bottom of this module.

Steps that need processors not implemented yet (``Identity``,
``ConstantFilter``, ``FeaturePermute``, ``LabelShuffle``, ``Choice``,
``TaskDispatch``) are kept as commented placeholders.

"""

from sdm.processing import (
    MeanImpute,
    Recipe,
    SigmaClip,
    StandardScale,
)


def default_regression_recipe() -> Recipe:
    """Return the default single-estimator regression recipe.

    Mirrors the original TabICLv2 regressor: mean imputation
    (``SimpleImputer``), standard scaling (``CustomStandardScaler``), and
    two-stage 4-sigma outlier clipping (``OutlierRemover``) on the features;
    the target is standard-scaled and its inverse maps predictions back to the
    original space. Commented lines mark processors not implemented yet.
    """
    return Recipe(
        features=[
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
            StandardScale(),
        ],
        output=[
            # Identity(),
        ],
    )


# End-state target, once the missing processors exist: a single
# task-aware recipe that serves both regression and classification. It cycles
# per-member normalization with ``Choice`` + ``n_estimators`` and
# dispatches the target and output roles per task with ``TaskDispatch``.
# ``TabICLv2.default_recipe()`` would then pick the regression branch, and a
# classifier the classification branch. Class-index-to-label decoding stays
# driver-side (argmax + CategoricalTensor categories), not an output step.
#
# def default_recipe() -> Recipe:
#     return Recipe(
#         features=[
#             MeanImpute(),
#             # ConstantFilter(),
#             StandardScale(epsilon=1e-6),
#             # norm options: none, power, quantile, quantile_rtdl, robust
#             Choice([Identity(), Quantile(output_distribution="normal")]),
#             SigmaClip(threshold=4.0),
#             FeaturePermute(method="latin"),
#         ],
#         target=[
#             TaskDispatch({
#                 classification: ClassShuffle(method="shift"),
#                 regression: StandardScale(),
#             }),
#         ],
#         output=[
#             TaskDispatch({
#                 classification: SoftmaxTemperature(temperature=0.9),
#                 regression: Identity(),
#             }),
#         ],
#     )
