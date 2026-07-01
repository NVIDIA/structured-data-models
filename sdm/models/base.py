from abc import ABC, abstractmethod

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor


class BaseModel(torch.nn.Module, ABC):
    @torch.inference_mode()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if isinstance(x, TableTensor):
            invalid_columns = x.size(-1) - x.numerical.size(-1)
            if invalid_columns > 0:
                raise ValueError(
                    f"Expected 'x' to only hold numerical columns"
                    f"(got {invalid_columns} non-numerical "
                    f"{'column' if invalid_columns == 1 else 'columns'})"
                )
            x = x.numerical

        if isinstance(y, TableTensor):
            if y.size(-1) != 1:
                raise ValueError(
                    f"Expected 'y' to refer to a single column "
                    f"(got {y.size(-1)} columns)"
                )

            if y.categorical.numel() > 0:
                y = y.categorical
                if isinstance(y, CategoricalTensor):
                    y = y.as_tensor()
            else:
                assert y.numerical.numel() > 0
                y = y.numerical

        if x.dim() == y.dim() and y.size(-1) == 1:
            y = y.squeeze(-1)

        if x.size()[:-2] != y.size()[:-1]:
            raise ValueError(
                f"Expected 'x' and 'y' to share the same batch dimensions "
                f"(got {tuple(x.size()[:-2])} and {tuple(y.size()[:-1])}"
            )

        return self._forward(x, y)

    def fit(
        self,
        x: Tensor,
        y: Tensor,
    ) -> None:
        raise NotImplementedError

    def predict(
        self,
        x: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        pass
