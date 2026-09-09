"""V11 regression target mapping and median readout."""

import torch

import sdm
import sdm.processing as sp


class TargetQuantile(sp.Processor, sp.InvertibleMixin):
    """Fit the V11 normal-quantile mapping and invert the predicted median."""

    handles_stypes = frozenset({sdm.Stype.numerical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.mapping = sp.QuantileTransform(
            n_quantiles=1000,
            subsample=100_000,
            output_distribution="normal",
        )
        self.register_buffer("lower", torch.empty(0))
        self.register_buffer("upper", torch.empty(0))

    def _fit(
        self,
        table: sdm.TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        # Contexts have at most 10k rows, below the V11 100k subsample cap.
        self.mapping.fit(table, generator=generator)
        self.lower = table.numerical.amin(dim=-2, keepdim=True)
        self.upper = table.numerical.amax(dim=-2, keepdim=True)

    def _transform(self, table: sdm.TableTensor) -> sdm.TableTensor:
        output = self.mapping.transform(table)
        normal = torch.distributions.Normal(0.0, 1.0)
        epsilon = table.numerical.new_tensor(
            1e-7 - torch.finfo(torch.float64).eps
        )
        numerical = torch.where(
            table.numerical <= self.lower,
            normal.icdf(epsilon),
            output.numerical,
        )
        numerical = torch.where(
            table.numerical >= self.upper,
            normal.icdf(1 - epsilon),
            numerical,
        )
        return output.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: sdm.TableTensor) -> sdm.TableTensor:
        if "q500" in table.columns[sdm.Stype.numerical]:
            table = table[["q500"]]
        return self.mapping.inverse_transform(table)
