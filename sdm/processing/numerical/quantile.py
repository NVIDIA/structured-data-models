from typing import Literal, cast

import torch
from torch import Tensor

from sdm.processing.ensemble import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor

BOUNDS_THRESH = 1e-7
_MAX_NUM_COLS = 32


def _torch_interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    n = xp.numel()
    if n == 1:
        return fp[0].expand_as(x)

    idx = torch.searchsorted(xp, x, right=True).clamp(1, n - 1)

    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]

    denom = x1 - x0
    weight = torch.where(denom != 0, (x - x0) / denom, 0.0)
    result = torch.lerp(y0, y1, weight)

    result = torch.where(x <= xp[0], fp[0], result)
    return torch.where(x >= xp[-1], fp[-1], result)


def _batched_interp(
    values: Tensor,
    boundaries: Tensor,
    references: Tensor,
) -> Tensor:
    # Batched 1D linear interpolation over independent feature grids:
    # values ``[F, N]`` with boundaries/references as ``[Q]`` or ``[F, Q]``.
    n = boundaries.shape[-1]
    if n == 1:
        if references.dim() == 1:
            return references[0].expand_as(values)
        return references[:, :1].expand_as(values)

    idx = torch.searchsorted(boundaries, values, right=True).clamp(1, n - 1)

    if boundaries.dim() == 1:
        x0 = boundaries[idx - 1]
        x1 = boundaries[idx]
        lower_boundary = boundaries[0]
        upper_boundary = boundaries[-1]
    else:
        x0 = boundaries.gather(1, idx - 1)
        x1 = boundaries.gather(1, idx)
        lower_boundary = boundaries[:, :1]
        upper_boundary = boundaries[:, -1:]

    if references.dim() == 1:
        y0 = references[idx - 1]
        y1 = references[idx]
        lower = references[0]
        upper = references[-1]
    else:
        y0 = references.gather(1, idx - 1)
        y1 = references.gather(1, idx)
        lower = references[:, :1]
        upper = references[:, -1:]

    denom = x1 - x0
    weight = torch.where(denom != 0, (values - x0) / denom, 0.0)
    result = torch.lerp(y0, y1, weight)

    result = torch.where(values <= lower_boundary, lower, result)
    return torch.where(values >= upper_boundary, upper, result)


class _QuantileState(torch.nn.Module):
    """Store fitted empirical quantiles."""

    quantiles: Tensor
    references: Tensor

    def __init__(self, quantiles: Tensor, references: Tensor) -> None:
        super().__init__()
        self.register_buffer("quantiles", quantiles, persistent=False)
        self.register_buffer("references", references, persistent=False)


class QuantileTransform(EnsembleProcessor, EnsembleInvertibleMixin):
    """Map numerical columns through their empirical quantiles.

    QuantileTransform grids are capped by the number of fitted rows and, when
    ``subsample`` is set, by ``20%`` of the subsample size to keep dense grids
    tractable.

    The subsample rows are drawn from the ``generator`` passed to ``fit()``;
    without one, they are drawn from the data device's global generator.

    Args:
        n_quantiles: Maximum number of quantiles to compute.
        subsample: Maximum number of rows to use for quantile computation.
        output_distribution: Distribution to map the empirical quantiles to.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        subsample: int | None = 10_000,
        output_distribution: Literal["uniform", "normal"] = "uniform",
    ) -> None:
        super().__init__()
        if n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive.")
        if subsample is not None and subsample <= 0:
            raise ValueError("subsample must be positive or None.")
        self._n_quantiles = n_quantiles
        self.subsample = subsample
        self.output_distribution = output_distribution
        self._states = torch.nn.ModuleList()
        self._state_ids: tuple[int, ...] = ()

    def _subsample_indices(
        self,
        inp: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        return torch.randperm(
            inp.shape[0],
            generator=generator,
            device=inp.device,
        )[: self.subsample]

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        first_table = ensemble_table.table(0)
        share_state = (
            self.subsample is None or self.subsample >= first_table.size(-2)
        ) and sum(group.size(0) for group in ensemble_table) == 1
        num_states = 1 if share_state else ensemble_table.num_members

        # TODO: For mixed ensembles, reuse deterministic states and outputs
        # once EnsembleTable exposes public logical table identities.
        states = []
        for state_id in range(num_states):
            table = ensemble_table.table(0 if share_state else state_id)
            numerical = table.numerical
            n_samples = numerical.shape[0]
            quantile_limit = n_samples
            if self.subsample is not None:
                # Keep quantiles well below the subsample size; very dense
                # percentile grids are slow to fit and add little resolution.
                quantile_limit = min(
                    quantile_limit,
                    int(self.subsample * 0.2),
                )
            n_quantiles = max(
                1,
                min(self._n_quantiles, quantile_limit),
            )
            references = torch.linspace(
                0,
                1,
                n_quantiles,
                device=numerical.device,
                dtype=numerical.dtype,
            )

            if self.subsample is not None and self.subsample < n_samples:
                indices = self._subsample_indices(numerical, generator)
                input_sample = numerical[indices]
            else:
                input_sample = numerical

            states.append(
                _QuantileState(
                    torch.quantile(
                        input_sample,
                        references,
                        dim=0,
                    ),
                    references,
                )
            )

        self._states = torch.nn.ModuleList(states)
        self._state_ids = (
            (0,) * ensemble_table.num_members
            if share_state
            else tuple(range(ensemble_table.num_members))
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._apply_ensemble(ensemble_table, inverse=False)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._apply_ensemble(ensemble_table, inverse=True)

    def _apply_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        inverse: bool,
    ) -> EnsembleTable:
        if len(self._state_ids) != ensemble_table.num_members:
            raise RuntimeError(
                f"{self.__class__.__name__!r} was fitted with "
                f"{len(self._state_ids)} ensemble members, but got "
                f"{ensemble_table.num_members}."
            )

        if (
            len(self._states) == 1
            and sum(group.size(0) for group in ensemble_table) == 1
        ):
            state = cast(_QuantileState, self._states[0])
            table = ensemble_table.table(0)
            output = (
                self._inverse_transform_with_state(table, state)
                if inverse
                else self._transform_with_state(table, state)
            )
            return EnsembleTable(
                output,
                num_members=ensemble_table.num_members,
            )

        tables = []
        for member_id, state_id in enumerate(self._state_ids):
            state = cast(_QuantileState, self._states[state_id])
            table = ensemble_table.table(member_id)
            tables.append(
                self._inverse_transform_with_state(table, state)
                if inverse
                else self._transform_with_state(table, state)
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )

    def _transform_with_state(
        self,
        table: TableTensor,
        state: _QuantileState,
    ) -> TableTensor:
        numerical = table.numerical
        transformed = torch.empty_like(numerical)
        for start in range(0, numerical.shape[1], _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, numerical.shape[1])
            # Searchsorted works over the innermost dimension, so columns
            # become independent rows: input ``[N, F]`` -> ``[F, N]``.
            input_columns = numerical[:, start:end].T.contiguous()
            quantile_columns = state.quantiles[:, start:end].T.contiguous()
            lower_bound_x = quantile_columns[:, :1]
            upper_bound_x = quantile_columns[:, -1:]
            if self.output_distribution == "normal":
                bounds_thresh = input_columns.new_tensor(BOUNDS_THRESH)
                lower_bounds_idx = (
                    input_columns - bounds_thresh < lower_bound_x
                )
                upper_bounds_idx = (
                    input_columns + bounds_thresh > upper_bound_x
                )
            elif self.output_distribution == "uniform":
                lower_bounds_idx = input_columns == lower_bound_x
                upper_bounds_idx = input_columns == upper_bound_x
            else:
                raise ValueError(
                    "output_distribution must be 'uniform' or 'normal'."
                )

            finite = input_columns.isfinite()
            forward = _batched_interp(
                input_columns,
                quantile_columns,
                state.references,
            )
            backward = _batched_interp(
                -input_columns,
                -quantile_columns.flip(1),
                -state.references.flip(0),
            )
            output = 0.5 * (forward - backward)

            output = torch.where(finite, output, input_columns)
            output = torch.where(upper_bounds_idx, 1.0, output)
            output = torch.where(lower_bounds_idx, 0.0, output)

            if self.output_distribution == "normal":
                eps = input_columns.new_tensor(
                    BOUNDS_THRESH - torch.finfo(torch.float64).eps
                )
                output = torch.special.ndtri(output)
                clip_min = torch.special.ndtri(eps)
                clip_max = torch.special.ndtri(1.0 - eps)
                output = output.clamp(clip_min, clip_max)

            transformed[:, start:end] = output.T.contiguous()
        return table.replace_blocks(numerical=transformed)

    def _inverse_transform_with_state(
        self,
        table: TableTensor,
        state: _QuantileState,
    ) -> TableTensor:
        numerical = table.numerical
        inverse = torch.empty_like(numerical)
        for start in range(0, numerical.shape[1], _MAX_NUM_COLS):
            end = min(start + _MAX_NUM_COLS, numerical.shape[1])
            # Searchsorted works over the innermost dimension, so columns
            # become independent rows: input ``[N, F]`` -> ``[F, N]``.
            input_columns = numerical[:, start:end].T.contiguous()
            quantile_columns = state.quantiles[:, start:end].T.contiguous()
            lower_bound_y = quantile_columns[:, :1]
            upper_bound_y = quantile_columns[:, -1:]
            if self.output_distribution == "normal":
                input_columns = torch.special.ndtr(input_columns)

            if self.output_distribution == "normal":
                bounds_thresh = input_columns.new_tensor(BOUNDS_THRESH)
                lower_bounds_idx = input_columns - bounds_thresh < 0.0
                upper_bounds_idx = input_columns + bounds_thresh > 1.0
            elif self.output_distribution == "uniform":
                lower_bounds_idx = input_columns == 0.0
                upper_bounds_idx = input_columns == 1.0
            else:
                raise ValueError(
                    "output_distribution must be 'uniform' or 'normal'."
                )

            finite = input_columns.isfinite()
            output = _batched_interp(
                input_columns,
                state.references,
                quantile_columns,
            )
            output = torch.where(finite, output, input_columns)
            output = torch.where(upper_bounds_idx, upper_bound_y, output)
            output = torch.where(lower_bounds_idx, lower_bound_y, output)
            inverse[:, start:end] = output.T.contiguous()
        return table.replace_blocks(numerical=inverse)
