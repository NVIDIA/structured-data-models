import sdm.processing as sp

MAX_ROWS = 1_000_000


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.ContextQueryDispatch(
                context=sp.SelectRows(MAX_ROWS, method="round_robin"),
            ),
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value", min_frequency=2),
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.ImputeMean(),
                    sp.DropConstantColumns(),
                    sp.Standardize(epsilon=1e-6),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.Choice(
                        sp.Identity(),
                        sp.PowerTransform(),
                        method="round_robin",
                    ),
                    sp.ClipSigma(threshold=4.0),
                    sp.FlipSign(),
                    sp.SelectColumns(500, method="round_robin"),
                    sp.ShuffleColumns(method="latin"),
                ],
            ),
        ],
        target=[
            sp.SelectRows(MAX_ROWS, method="round_robin"),
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(),
                ],
            ),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=0.9),
            ),
        ],
    )
