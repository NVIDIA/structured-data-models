import sdm.processing as sp


def default_recipe() -> sp.Recipe:  # noqa: D103
    return sp.Recipe(
        features=sp.StypeDispatch(
            datetime=[
                sp.AddCalendarFields(
                    fields=[
                        "minute",
                        "hour",
                        "weekday",
                        "day_of_month",
                        "month",
                    ],
                    encoding="cyclic",
                ),
                sp.DropStypes("datetime"),
            ],
        ),
        target=sp.Identity(),
        output=sp.ReduceEstimators(method="mean"),
    )
