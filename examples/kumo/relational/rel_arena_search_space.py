"""Shared RelArena policy: text OFF or context-fitted PCA32."""

from relarena.search_space import SearchSpace

DEFAULT = {
    "num_neighbors": [16, 16],
    "text_pca_components": 0,
    "regression_transform": "standard",
    "cache_text": True,
    "text_cache_max_bytes": 32 * 1024**3,
    "recent_context_pool": False,
    "max_keys": 20_000,
    "entity_timestamps": "retain",
    "temporal_strategy": "last",
    "raw_event_lags": True,
}

# Default first: the two candidates differ only in whether text is used.
SEARCH_SPACE = SearchSpace(
    default_overrides=DEFAULT,
    fixed_grid=[
        {**DEFAULT, "text_pca_components": components}
        for components in (0, 32)
    ],
)
