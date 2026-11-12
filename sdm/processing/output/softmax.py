# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from sdm import Stype, TableLike
from sdm.processing import Processor


class Softmax(Processor):
    """Apply temperature-scaled softmax over the final class dimension.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = False

    def __init__(
        self,
        *,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def _transform(self, table: TableLike) -> TableLike:
        numerical = (table.numerical / self.temperature).softmax(dim=-1)
        return table.replace_blocks(numerical=numerical)

    def __repr__(self, *, indent: int = 0) -> str:
        if self.temperature == 1.0:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"temperature={self.temperature})"
        )
