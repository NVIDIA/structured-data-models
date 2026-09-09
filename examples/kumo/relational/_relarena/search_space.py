"""Shared development candidates; complete 24-hour runtime is not certified."""

from examples.kumo.relational._relarena.adapter import DEFAULT
from relarena.search_space import SearchSpace

# Stopped V12's balanced design covers all pairs and triples of six factors.
# It is not an exhaustive Cartesian search or a finalized benchmark policy.
# Neighbors, raw lags, label lags, text PCA, targets, recency, entity dates.
_CHOICES = [
    ([16, 16], False, 0, 0, "standard", False, "retain"),
    ([1], False, 0, 0, "quantile", True, "drop"),
    ([1], False, 0, 32, "standard", False, "retain"),
    ([1], True, 0, 0, "standard", False, "drop"),
    ([1], True, 0, 32, "quantile", True, "retain"),
    ([1], False, 2, 0, "quantile", False, "retain"),
    ([1], False, 2, 32, "standard", True, "drop"),
    ([1, 1], False, 0, 0, "standard", True, "drop"),
    ([1, 1], False, 0, 32, "quantile", False, "retain"),
    ([1, 1], True, 0, 0, "quantile", False, "drop"),
    ([1, 1], True, 0, 32, "standard", True, "retain"),
    ([1, 1], False, 2, 0, "standard", False, "retain"),
    ([1, 1], False, 2, 32, "quantile", True, "drop"),
    ([16], False, 0, 0, "quantile", False, "drop"),
    ([16], False, 0, 32, "standard", True, "retain"),
    ([16], True, 0, 0, "standard", False, "retain"),
    ([16], True, 0, 32, "quantile", True, "drop"),
    ([16], False, 2, 0, "quantile", True, "retain"),
    ([16], False, 2, 32, "standard", False, "drop"),
    ([16, 16], False, 0, 32, "quantile", True, "drop"),
    ([16, 16], True, 0, 0, "quantile", True, "retain"),
    ([16, 16], True, 0, 32, "standard", False, "drop"),
    ([16, 16], False, 2, 0, "standard", True, "drop"),
    ([16, 16], False, 2, 32, "quantile", False, "retain"),
    ([64, 64], False, 0, 0, "quantile", True, "drop"),
    ([64, 64], False, 0, 32, "standard", False, "retain"),
    ([64, 64], True, 0, 0, "standard", True, "retain"),
    ([64, 64], True, 0, 32, "quantile", False, "drop"),
    ([64, 64], False, 2, 0, "quantile", False, "retain"),
    ([64, 64], False, 2, 32, "standard", True, "drop"),
]

SEARCH_SPACE = SearchSpace(
    default_overrides=DEFAULT,
    fixed_grid=[
        {
            **DEFAULT,
            "num_neighbors": neighbors,
            "raw_event_lags": raw,
            "history_lags": history,
            "text_pca_components": text,
            "regression_transform": targets,
            "recent_context_pool": recent,
            "entity_timestamps": dates,
        }
        for neighbors, raw, history, text, targets, recent, dates in _CHOICES
    ],
)
