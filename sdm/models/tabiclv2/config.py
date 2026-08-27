from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TabICLv2InferenceConfig:
    r"""Advanced controls for TabICLv2 inference.

    Args:
        row_chunk_size: Maximum rows processed together by the memory-efficient
            row-embedding path.
        column_chunk_size: Maximum columns processed together while building
            the memory-efficient row-embedding summaries.
    """

    row_chunk_size: int = 2048
    column_chunk_size: int = 4
