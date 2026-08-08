import sdm
import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sdm.models.TabICLv2.default_recipe().prepend_features(
        sp.StypeDispatch(
            datetime=sp.AddCalendarFields(
                fields=("minute", "hour", "weekday", "day_of_month", "month"),
            )
        )
    )
