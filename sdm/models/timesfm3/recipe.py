import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        output=sp.ReduceEstimators(method="mean"),
    )
