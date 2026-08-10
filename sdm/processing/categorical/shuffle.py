from typing import Literal, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable


class _CategoryPermutations(torch.nn.Module):
    """Store fitted categorical code permutations."""

    permutations: Tensor
    offsets: Tensor

    def __init__(
        self,
        permutations: Tensor,
        offsets: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.register_buffer("permutations", permutations)
        self.register_buffer(
            "offsets",
            torch.tensor(
                offsets,
                dtype=torch.long,
                device=permutations.device,
            ),
        )


class ShuffleCategories(EnsembleProcessor):
    """Independently permute the integer codes of categorical columns.

    One permutation per categorical column is drawn when the processor is
    fitted; pass ``generator`` to ``fit()`` to make the draws reproducible.
    For an ensemble, each logical member receives independent permutations.
    Codes and their corresponding category vectors are permuted together so
    decoded values remain unchanged. Negative codes represent missing values
    and are preserved unchanged. Only categorical columns are supported; use
    :class:`~sdm.processing.StypeDispatch` to apply this processor to the
    categorical block of a mixed feature table.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            codes by a drawn offset, and ``"random"`` remaps the codes with
            a drawn permutation.
    """

    operates_on_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self._permutations = torch.nn.ModuleList()
        self._permutation_ids: tuple[int, ...] = ()

    def _draw_permutations(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, tuple[int, ...]]:
        device = table.categorical.device
        permutations: list[Tensor] = []
        offsets = [0]
        for category in table.categorical.categories:
            n_classes = category.numel()
            if n_classes <= 1:
                permutation = torch.arange(n_classes, device=device)
            elif self.method == "shift":
                offset = torch.randint(
                    n_classes,
                    (1,),
                    generator=generator,
                    device=device,
                )
                permutation = (
                    torch.arange(n_classes, device=device) - offset
                ) % n_classes
            else:
                assert self.method == "random"
                permutation = torch.randperm(
                    n_classes,
                    generator=generator,
                    device=device,
                )
            permutations.append(permutation)
            offsets.append(offsets[-1] + n_classes)

        permutation = (
            torch.cat(permutations)
            if len(permutations) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )
        return permutation, tuple(offsets)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        self._permutations = torch.nn.ModuleList()
        permutation_ids = []
        permutation_id_by_key: dict[
            tuple[torch.device, tuple[int, ...], tuple[int, ...]], int
        ] = {}

        for member_id in range(ensemble_table.num_members):
            permutations, offsets = self._draw_permutations(
                ensemble_table.table(member_id),
                generator=generator,
            )
            key = (
                permutations.device,
                offsets,
                tuple(permutations.tolist()),
            )
            permutation_id = permutation_id_by_key.get(key)
            if permutation_id is None:
                permutation_id = len(self._permutations)
                permutation_id_by_key[key] = permutation_id
                self._permutations.append(
                    _CategoryPermutations(permutations, offsets)
                )
            permutation_ids.append(permutation_id)

        self._permutation_ids = tuple(permutation_ids)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutation_ids) != ensemble_table.num_members:
            raise RuntimeError(
                "ShuffleCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        member_ids_by_permutation: dict[int, list[int]] = {}
        for member_id, permutation_id in enumerate(self._permutation_ids):
            member_ids_by_permutation.setdefault(permutation_id, []).append(
                member_id
            )

        if len(member_ids_by_permutation) == ensemble_table.num_members:
            member_tables: list[TableTensor] = []
            for member_id, permutation_id in enumerate(self._permutation_ids):
                permutations = cast(
                    _CategoryPermutations,
                    self._permutations[permutation_id],
                )
                member_tables.append(
                    self._permute(
                        ensemble_table.table(member_id),
                        permutations,
                    )
                )
            return EnsembleTable.from_tables(
                tables=member_tables,
                member_table_ids=range(ensemble_table.num_members),
            )

        outputs: dict[int, EnsembleTable] = {}
        for permutation_id, member_ids in member_ids_by_permutation.items():
            selected = ensemble_table.select_members(member_ids)
            permutations = cast(
                _CategoryPermutations,
                self._permutations[permutation_id],
            )
            outputs[permutation_id] = selected.replace_groups(
                [self._permute(group, permutations) for group in selected]
            )

        output_tables = []
        member_ids = []
        next_member_id_by_permutation: dict[int, int] = {}
        for permutation_id in self._permutation_ids:
            output_tables.append(outputs[permutation_id])
            member_id = next_member_id_by_permutation.get(permutation_id, 0)
            member_ids.append(member_id)
            next_member_id_by_permutation[permutation_id] = member_id + 1

        return EnsembleTable.gather_members(
            tables=output_tables,
            member_ids=member_ids,
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        permutations: _CategoryPermutations,
    ) -> TableTensor:
        offsets = permutations.offsets.tolist()
        code = table.categorical.code.clone()
        valid_mask = table.categorical.isfinite()
        categories: list[Tensor] = []
        for index, category in enumerate(table.categorical.categories):
            permutation = permutations.permutations[
                offsets[index] : offsets[index + 1]
            ]
            codes = code[..., index]
            valid = valid_mask[..., index]
            valid_codes = codes[valid].to(torch.long)
            codes[valid] = permutation[valid_codes].to(codes.dtype)
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
