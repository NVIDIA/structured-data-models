import sdm.processing as sp
from sdm import TableTensor


def _replace_inf(table: TableTensor) -> TableTensor:
    numerical = table.numerical
    return table.replace_blocks(
        numerical=numerical.masked_fill(numerical.isinf(), float("nan"))
    )


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value", min_frequency=2),
                    sp.ToNumerical(),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    _replace_inf,
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
                    sp.ShuffleColumns(method="latin"),
                    sp.SelectColumns(500, method="first"),
                ],
            ),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=[
                    sp.Standardize(),
                    sp.FlipSign(flip_order=True),
                ],
            ),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=1.0),
            ),
        ],
    )
