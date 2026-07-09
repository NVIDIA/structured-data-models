"""TabICLv2."""

from sdm.models.tabiclv2.model import TabICLv2
from sdm.models.tabiclv2.output import decode_regression_quantiles


__all__ = [
    "TabICLv2",
    "decode_regression_quantiles",
]
