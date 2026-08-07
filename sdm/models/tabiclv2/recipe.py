import sdm.processing as sp
from sdm import Recipe


def default_recipe() -> Recipe:  # noqa: D103
    return Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.ImputeMean(),
                    sp.DropConstantColumns(),
                    sp.Standardize(epsilon=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(sp.Identity(), sp.PowerTransform()),
                    sp.ClipSigma(threshold=4.0),
                    sp.ShuffleColumns(method="shift"),
                ],
            ),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=sp.Standardize(),
            ),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=0.9),
            ),
        ],
    )
