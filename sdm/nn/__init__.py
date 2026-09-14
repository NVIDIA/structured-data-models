"""Neural network modules for structured data models."""

from sdm.nn._cudnn_varlen import cudnn_varlen_stats, enable_cudnn_varlen
from sdm.nn.rope import RotaryEmbedding
from sdm.nn.glu import SwiGLU
from sdm.nn.softplus import SoftplusScale
from sdm.nn.scaling import QueryScaling, QASSMax, LogScale, GatedLogScale
from sdm.nn.attention import SDPA, Attention, TransformerBlock
from sdm.nn.set_transformer import InducedTransformerBlock


__all__ = [
    "cudnn_varlen_stats",
    "enable_cudnn_varlen",
    "RotaryEmbedding",
    "SwiGLU",
    "SoftplusScale",
    "QueryScaling",
    "QASSMax",
    "LogScale",
    "GatedLogScale",
    "SDPA",
    "Attention",
    "TransformerBlock",
    "InducedTransformerBlock",
]
