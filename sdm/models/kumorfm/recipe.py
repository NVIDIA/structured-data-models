from sdm.models.tabiclv2.recipe import default_recipe as tabiclv2_recipe
from sdm.processing import (
    AddCalendarFields,
    Recipe,
    StypeDispatch,
)


def default_recipe() -> Recipe:  # noqa: D103
    datetime_processor = StypeDispatch(
        datetime=AddCalendarFields(
            fields=("minute", "hour", "weekday", "day_of_month", "month"),
        )
    )
    recipe = tabiclv2_recipe()
    recipe.features = datetime_processor + recipe.features
    return recipe
