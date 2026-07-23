from typing import cast

from sdm import Stype
from sdm.models.tabiclv2.recipe import default_recipe as _default_recipe
from sdm.processing import (
    DispatchByStype,
    EncodeCalendar,
    Recipe,
    Sequential,
)


def default_recipe() -> Recipe:  # noqa: D103
    recipe = _default_recipe()
    dispatch_by_stype = cast(Sequential, recipe.features).steps[0]
    assert isinstance(dispatch_by_stype, DispatchByStype)
    dispatch_by_stype.processors[Stype.datetime.value] = EncodeCalendar(
        features=(
            "minute",
            "hour",
            "weekday",
            "day_of_month",
            "month",
        )
    )
    return recipe
