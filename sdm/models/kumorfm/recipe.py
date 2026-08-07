import sdm.processing as sp
from sdm.models.tabiclv2.recipe import default_recipe as tabiclv2_recipe


def default_recipe() -> sp.Recipe:  # noqa: D103
    recipe = tabiclv2_recipe()
    recipe.features = (
        sp.TableDispatch(
            related=sp.StypeDispatch(
                datetime=sp.AddCalendarFields(
                    fields=(
                        "minute",
                        "hour",
                        "weekday",
                        "day_of_month",
                        "month",
                    ),
                ),
            ),
        )
        + recipe.features
    )
    return recipe
