import sdm.processing as sp


def default_recipe() -> sp.Recipe:
    """Return the minimal Kumo Tabular processing recipe.

    The feature pipeline aligns categorical codes between context and query,
    then places numerical columns before categorical columns in the dense
    model input. It expects otherwise preprocessed, finite features.
    """
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=sp.AlignCategories(sort_by="value"),
            ),
            sp.ToNumerical(),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.Softmax(),
        ],
    )
