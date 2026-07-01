import warnings
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor


class BaseModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`BaseModel` defines the public inferface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train] or [..., R_train, 1]
    ) -> Tensor:  # [..., R - R_test, *]
        r"""The in-context learning forward pass.

        Args:
            x: The feature tensor with shape ``[..., R, C]`` with ``R`` rows
                and ``C`` columns.
                The first ``R_train`` rows along ``R`` refer to the in-context
                examples.
                Features may be given as :class:`torch.Tensor` or
                :class:`~sdm.tensor.TableTensor` instances.
            y: The targets of in-context examples with shape
                ``[..., R_train]`` or ``[..., R_train, 1]``.
                Targets may be given as :class:`torch.Tensor` or
                :class:`~sdm.tensor.TableTensor` instances.

        Returns:
            The prediction for the remaining ``[..., R - R_train]`` test rows.
        """
        if isinstance(x, TableTensor):
            invalid_columns = x.size(-1) - x.numerical.size(-1)
            if invalid_columns > 0:
                invalid_stypes = [
                    stype.value
                    for stype, tensor in x.items()
                    if tensor.size(-1) > 0 and stype not in (Stype.numerical,)
                ]
                warnings.warn(
                    f"Expected 'x' to only hold numerical columns but also "
                    f"found {'/'.join(invalid_stypes)} data. "
                    f"This data will be ignored. "
                    f"Make sure that your recipe converts such types to "
                    f"numerical data to include them as features.",
                    stacklevel=2,
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

    @torch.inference_mode()
    def fit(
        self,
        x: Tensor,  # [..., R_train, C]
        y: Tensor,  # [..., R_train] or [..., R_train, 1]
    ) -> None:
        r"""Fit and cache in-context examples.

        Repeated calls to :meth:`predict` can then reuse the same in-context
        examples while only providing new test rows.

        Args:
            x: The feature tensor with shape ``[..., R_train, C]`` with
                ``R_train`` rows and ``C`` columns.
                Features may be given as :class:`torch.Tensor` or
                :class:`~sdm.tensor.TableTensor` instances.
            y: The targets of in-context examples with shape
                ``[..., R_train]`` or ``[..., R_train, 1]``.
                Targets may be given as :class:`torch.Tensor` or
                :class:`~sdm.tensor.TableTensor` instances.
        """
        # TODO Implement real key/value caching.
        self.clear()
        self._x = x
        self._y = y

    def clear(self) -> None:
        r"""Clears cached in-context examples."""
        # TODO Implement real key/value caching.
        if hasattr(self, "_x"):
            del self._x
        if hasattr(self, "_y"):
            del self._y

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor,  # [..., R_test, C]
    ) -> Tensor:  # [..., R_test, *]
        r"""Predict unseen test examples.

        .. note::

            This method requires a prior call to :meth:`fit`.

        Args:
            x: The feature tensor with shape ``[..., R_test, C]`` with
                ``R_test`` rows and ``C`` columns.
                Features may be given as :class:`torch.Tensor` or
                :class:`~sdm.tensor.TableTensor` instances.

        Returns:
            The prediction for ``[..., R_test]`` test rows.
        """
        if not hasattr(self, "_x") or not hasattr(self, "_y"):
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
                f"'{self.__class__.__name__}.fit()' beforehand."
            )

        return self.forward(
            x=torch.cat([self._x, x], dim=-2),
            y=self._y,
        )

    # Abstract Methods ########################################################

    @abstractmethod
    def _forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
    ) -> Tensor:  # [..., R - R_train, *]
        pass
