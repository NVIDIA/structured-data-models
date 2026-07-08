from sdm.processing import (
    Clamp,
    Identity,
    MeanImpute,
    Processor,
    Recipe,
    SigmaClip,
    StandardScale,
)


def _feature_processors() -> list[Processor]:
    return [
        MeanImpute(),
        StandardScale(epsilon=1e-6),
        Clamp(min_value=-100.0, max_value=100.0),
        SigmaClip(threshold=4.0),
    ]


def default_classification_recipe() -> Recipe:
    """Return the default single-estimator TabFM classification recipe.

    Numerical features follow the upstream ``"none"`` normalization path:
    mean imputation, standard scaling with epsilon, fixed clipping, and
    two-stage sigma clipping. Categorical indices and their masks must bypass
    this numerical feature pipeline.

    Returns:
        A :class:`~sdm.processing.Recipe` with an identity target transform.
    """
    return Recipe(
        features=_feature_processors(),
        target=[Identity()],
    )


def default_regression_recipe() -> Recipe:
    """Return the default single-estimator TabFM regression recipe.

    Numerical features use the same context-fitted path as classification.
    Regression targets are standard-scaled on context rows and predictions are
    mapped back through the inverse target transform. Categorical indices and
    their masks must bypass the numerical feature pipeline.

    Returns:
        A :class:`~sdm.processing.Recipe` with an invertible target scaler.
    """
    return Recipe(
        features=_feature_processors(),
        target=[StandardScale()],
    )
