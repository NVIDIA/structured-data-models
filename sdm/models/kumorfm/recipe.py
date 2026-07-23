from typing import cast

from sdm import Stype
from sdm.models.tabiclv2.recipe import default_recipe as _default_recipe
from sdm.processing import (
    CalendarParts,
    Recipe,
    Sequential,
    StypeDispatch,
)


def default_recipe() -> Recipe:  # noqa: D103
    recipe = _default_recipe()
    stype_dispatch = cast(Sequential, recipe.features).steps[0]
    assert isinstance(stype_dispatch, StypeDispatch)
    stype_dispatch.processors[Stype.datetime.value] = CalendarParts(
        features=(
            "minute",
            "hour",
            "weekday",
            "day_of_month",
            "month",
        )
    )
    return recipe
