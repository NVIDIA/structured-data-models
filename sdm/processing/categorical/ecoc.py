# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import Tensor

from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor


def ecoc_code_count(num_classes: int, alphabet_size: int) -> int:
    """Return the number of distinct codes a many-class target needs.

    One code separates ``alphabet_size - 1`` classes, so ``cover`` codes
    reach every class once. Small class counts get more codes than that,
    because error correction needs several votes per class.
    """
    cover = math.ceil(num_classes / (alphabet_size - 1))
    depth = math.ceil(math.log(max(num_classes, 2), alphabet_size))
    return max(cover, 4 * depth)


class ECOCCategories(EnsembleProcessor):
    """Merge a many-class target into a small alphabet, one code per member.

    A model with a fixed class capacity cannot fit a target that holds more
    categories than that capacity. This processor gives every ensemble member
    a different many-to-few relabeling, so each member solves a task the model
    can represent. The last symbol is the "rest" symbol, which collects the
    classes a member does not separate.

    The processor is the identity while the target holds at most
    ``alphabet_size`` categories, so ordinary targets are unchanged.

    The number of distinct codes follows the class count. Members beyond that
    number repeat the codes, so a member count that is a multiple of the code
    count gives every code several feature-diverse members.

    :class:`~sdm.processing.DecodeECOC` reads the fitted codebook and maps the
    member outputs back to the original classes.

    Args:
        alphabet_size: The number of symbols the model can represent.
    """

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(self, alphabet_size: int) -> None:
        super().__init__()
        if alphabet_size < 3:
            raise ValueError("alphabet_size must be at least 3")
        self.alphabet_size = alphabet_size
        self.register_buffer("codebook", torch.empty(0, dtype=torch.long))
        self._categories: BufferList[Tensor] = BufferList()

    @property
    def rest(self) -> int:
        """Return the symbol that collects the classes a member merges."""
        return self.alphabet_size - 1

    @property
    def active(self) -> bool:
        """Return whether the target needs more symbols than the alphabet."""
        return self.codebook.numel() > 0

    @property
    def categories(self) -> Tensor:
        """Return the original categories the codebook was fitted on."""
        return self._categories[0]

    def _draw_codebook(
        self,
        num_classes: int,
        num_codes: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Draw a codebook that holds the classes as far apart as it can.

        Every code separates the classes that the earlier codes covered
        least, and the jitter breaks ties. The best of several draws wins,
        scored by the distance between the columns of the codebook.
        """
        device = torch.device("cpu") if generator is None else generator.device
        best: Tensor | None = None
        best_score = (-1, -1.0)
        for _ in range(50):
            codebook = torch.full(
                size=(num_codes, num_classes),
                fill_value=self.rest,
                dtype=torch.long,
                device=device,
            )
            coverage = torch.zeros(num_classes, device=device)
            for code_id in range(num_codes):
                priority = coverage + 0.1 * torch.rand(
                    num_classes,
                    generator=generator,
                    device=device,
                )
                chosen = priority.argsort()[: min(self.rest, num_classes)]
                codebook[code_id, chosen] = torch.randperm(
                    self.rest,
                    generator=generator,
                    device=device,
                )[: chosen.numel()]
                coverage[chosen] += 1
            if not bool(coverage.all()):
                continue

            # Accumulated one code at a time, because holding every class
            # pair and every code at once grows with the cube of the classes.
            distance = torch.zeros(
                size=(num_classes, num_classes),
                dtype=torch.long,
                device=device,
            )
            for code in codebook:
                distance += code[:, None] != code[None, :]
            rows, cols = torch.triu_indices(
                num_classes,
                num_classes,
                offset=1,
                device=device,
            )
            pairwise = distance[rows, cols].float()
            score = (int(pairwise.min()), float(pairwise.mean()))
            if score > best_score:
                best, best_score = codebook, score

        if best is None:
            raise RuntimeError("Failed to generate a many-class codebook")
        return best

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        table = ensemble_table.table(0)
        if len(table.categorical.categories) != 1:
            raise ValueError(
                "'ECOCCategories' supports exactly one categorical target"
            )
        categories = table.categorical.categories[0]
        num_classes = categories.numel()
        num_members = ensemble_table.num_members

        if num_classes <= self.alphabet_size:
            self.codebook = torch.empty(0, dtype=torch.long)
            self._categories = BufferList()
            return

        # One codebook maps the classes of every member, so a member that
        # orders its categories differently would get the wrong symbols.
        for member_id in range(1, num_members):
            member = ensemble_table.table(member_id)
            if not torch.equal(member.categorical.categories[0], categories):
                raise ValueError(
                    "'ECOCCategories' needs the same target categories in "
                    f"every ensemble member (member {member_id} differs)"
                )

        minimum = math.ceil(num_classes / self.rest)
        if num_members < minimum:
            raise ValueError(
                f"'ECOCCategories' needs at least {minimum} estimators to "
                f"cover {num_classes} classes with alphabet size "
                f"{self.alphabet_size} (got {num_members})"
            )

        codebook = self._draw_codebook(
            num_classes=num_classes,
            num_codes=min(
                ecoc_code_count(num_classes, self.alphabet_size),
                num_members,
            ),
            generator=generator,
        )
        # Repeated up to the member count, so the decoder reads one code per
        # member and several members can share a code.
        repeat = torch.arange(
            num_members, device=codebook.device
        ) % codebook.size(0)
        self.codebook = codebook[repeat].to(categories.device)
        self._categories = BufferList([categories])

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not self.active:
            return ensemble_table
        if self.codebook.size(0) != ensemble_table.num_members:
            raise RuntimeError(
                "'ECOCCategories' must be fitted with the same number of "
                "ensemble members before transform"
            )

        tables: list[TableTensor] = []
        for member_id in range(ensemble_table.num_members):
            table = ensemble_table.table(member_id)
            code = table.categorical.code
            lookup = self.codebook[member_id].to(code.device)
            tables.append(
                table.replace_blocks(
                    categorical=CategoricalTensor(
                        code=torch.where(
                            table.categorical.isfinite(),
                            lookup[code.clamp_min(0).to(torch.long)].to(
                                code.dtype
                            ),
                            code,
                        ),
                        # Held complete, so every member reports the same
                        # output columns whichever symbols it happens to use.
                        categories=[
                            torch.arange(
                                self.alphabet_size,
                                device=code.device,
                            )
                        ],
                    ),
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(ensemble_table.num_members),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}"
            f"(alphabet_size={self.alphabet_size})"
        )
