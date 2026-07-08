import warnings
from abc import ABC, abstractmethod
from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.processing import InvertibleMixin, Recipe


class BaseModel(torch.nn.Module, ABC):
    r"""Base model for in-context foundation models on structured data.

    :class:`BaseModel` defines the public inferface shared among in-context
    foundation models on structured data.
    It enriches models by unified pre-processing and post-processing routines,
    key/value caching, and ensembling.
    """

    def __init__(self) -> None:
        super().__init__()

        # Fitted state reused across 'predict()' calls, holding shared
        # metadata, the fitted recipe, and one key/value sub-cache per
        # ensemble member.
        self._cache: Cache | None = None

    @torch.inference_mode()
    def forward(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
    ) -> Tensor:  # [..., R - R_train, *]
        r"""The in-context learning forward pass.

        Args:
            x: The feature tensor with shape ``[..., R, C]`` with ``R`` rows
                and ``C`` columns.
                The first ``R_train`` rows along ``R`` refer to the in-context
                examples.
            y: The targets of in-context examples with shape
                ``[..., R_train]`` or ``[..., R_train, 1]``.
            recipe: The pre- and postprocessing recipe applied around the
                model. Feature steps are fitted on the in-context rows only
                and applied to all rows, target steps are fitted on ``y`` and
                inverted on predictions, and output steps are applied last.
                Requires ``x`` and ``y`` to be :class:`~sdm.TableTensor`
                inputs. If ``None``, no recipe is applied.
            num_estimators: The number of ensemble members ``E``.
                The forward pass runs once per member, and predictions are
                averaged across members.

        Returns:
            The prediction for the remaining ``[..., R - R_train]`` test rows.
        """
        if num_estimators < 1:
            raise ValueError(
                f"Expected 'num_estimators' to be a positive integer "
                f"(got {num_estimators})"
            )

        x, y = self._preprocess(x, y, recipe=recipe)
        # TODO Create an ensemble dimension to process across ensemble
        # members for better efficiency.
        outs: list[Tensor] = []
        for _ in range(num_estimators):
            outs.append(self._forward(x, y, cache=None))
        return self._postprocess(torch.stack(outs).mean(dim=0), recipe)

    @torch.inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,  # [..., R_train, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        *,
        recipe: Recipe | None = None,
        num_estimators: int = 1,
    ) -> None:
        r"""Fit and cache in-context examples.

        Repeated calls to :meth:`predict` can then reuse the same in-context
        examples while only providing new test rows.

        Args:
            x: The feature tensor with shape ``[..., R_train, C]`` with
                ``R_train`` rows and ``C`` columns.
            y: The targets of in-context examples with shape
                ``[..., R_train]`` or ``[..., R_train, 1]``.
            recipe: The pre- and postprocessing recipe applied around the
                model. Feature and target steps are fitted on the in-context
                examples, and the fitted state is reused by subsequent
                :meth:`predict` calls. Requires ``x`` and ``y`` to be
                :class:`~sdm.TableTensor` inputs. If ``None``, no recipe is
                applied.
            num_estimators: The number of ensemble members ``E``.
                In-context examples are fitted once per member, and subsequent
                :meth:`predict` calls average predictions across members.
        """
        if num_estimators < 1:
            raise ValueError(
                f"Expected 'num_estimators' to be a positive integer "
                f"(got {num_estimators})"
            )

        self.clear()
        x, y = self._preprocess(x, y, recipe=recipe)
        x = x[..., : y.size(-1), :]
        members: list[Cache] = []
        for _ in range(num_estimators):
            member = Cache()
            self._forward(x, y, cache=member)
            member.freeze()
            members.append(member)
        cache = Cache(
            {
                "y.dtype": y.dtype,
                "recipe": recipe,
                "members": tuple(members),
            }
        )
        cache.freeze()
        self._cache = cache

    def clear(self) -> None:
        r"""Clears cached in-context examples."""
        self._cache = None

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,  # [..., R_test, C]
    ) -> Tensor:  # [..., R_test, *]
        r"""Predict unseen test examples.

        .. note::

            This method requires a prior call to :meth:`fit`.
            A recipe passed to :meth:`fit` is reused to transform ``x`` and
            to postprocess predictions.

        Args:
            x: The feature tensor with shape ``[..., R_test, C]`` with
                ``R_test`` rows and ``C`` columns.

        Returns:
            The prediction for ``[..., R_test]`` test rows, averaged across
            ensemble members when fitted with ``num_estimators > 1``.
        """
        if self._cache is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' not yet fitted. Make sure to "
                f"'{self.__class__.__name__}.fit()' beforehand."
            )

        recipe = cast(Recipe | None, self._cache["recipe"])
        if recipe is not None:
            if not isinstance(x, TableTensor):
                raise ValueError(
                    f"Expected 'x' to be a 'TableTensor' when fitted with a "
                    f"'recipe' (got '{type(x).__name__}')"
                )
            # Recipe steps run in normal mode since 'TableTensor' does not
            # support structural ops on inference tensors:
            with torch.inference_mode(False):
                x = recipe.features.transform(x)

        y = torch.empty(
            (*x.size()[:-2], 0),
            dtype=cast(torch.dtype, self._cache["y.dtype"]),
            device=x.device,
        )
        x, y = self._preprocess(x, y)
        outs: list[Tensor] = []
        for member in cast(tuple[Cache, ...], self._cache["members"]):
            outs.append(self._forward(x, y, cache=member))
        return self._postprocess(torch.stack(outs).mean(dim=0), recipe)

    # Helpers #################################################################

    def _preprocess(
        self,
        x: Tensor | TableTensor,  # [..., R, C]
        y: Tensor | TableTensor,  # [..., R_train] or [..., R_train, 1]
        recipe: Recipe | None = None,
    ) -> tuple[Tensor, Tensor]:
        if recipe is not None:
            if not isinstance(x, TableTensor) or not isinstance(
                y, TableTensor
            ):
                raise ValueError(
                    f"Expected 'x' and 'y' to be a 'TableTensor' when "
                    f"'recipe' is given (got '{type(x).__name__}' and "
                    f"'{type(y).__name__}')"
                )

            # Recipe steps run in normal mode since 'TableTensor' does not
            # support structural ops on inference tensors:
            with torch.inference_mode(False):
                # Fit feature steps on the in-context rows only to avoid
                # leakage:
                recipe.features.fit(x[..., : y.size(-2), :])
                x = recipe.features.transform(x)
                y = recipe.target.fit_transform(y)

        if isinstance(x, TableTensor):
            invalid_columns = x.size(-1) - x.numerical.size(-1) - x.id.size(-1)
            if invalid_columns > 0:
                invalid_stypes = [
                    stype.value
                    for stype, tensor in x.items()
                    if tensor.size(-1) > 0
                    and stype not in (Stype.numerical, Stype.id)
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

        return x, y

    def _postprocess(
        self,
        out: Tensor,  # [..., R_test, *]
        recipe: Recipe | None,
    ) -> Tensor:  # [..., R_test, *]
        if recipe is None:
            return out

        if not isinstance(recipe.target, InvertibleMixin):
            raise ValueError(
                f"Expected the target steps of 'recipe' to support "
                f"'inverse_transform' to map predictions back to the "
                f"original target space "
                f"(got '{recipe.target.__class__.__name__}')"
            )

        # Recipe steps run in normal mode since 'TableTensor' does not
        # support structural ops on inference tensors. Cloning the prediction
        # moves it out of inference mode:
        with torch.inference_mode(False):
            table = TableTensor.from_tensor(out.clone())
            table = recipe.target.inverse_transform(table)
            table = recipe.output.transform(table)
            return table.numerical

    # Abstract Methods ########################################################

    @abstractmethod
    def _forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R - R_train, *]
        pass

    @abstractmethod
    def default_recipe(self) -> Recipe:
        r"""Return the default processing recipe for this model.

        Model subclasses must override this method to expose the model-specific
        preprocessing and postprocessing recipe.

        Returns:
            The :class:`~sdm.processing.Recipe` applied during pre- and
            postprocessing by default.
        """
