# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, EnsembleTable, Stype, TableTensor
from sdm.processing import AddLevelCounts, EnsembleProcessor

CATEGORIES = (torch.arange(60), torch.arange(3))


def _table(wide: torch.Tensor, narrow: torch.Tensor) -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("city", "kind")},
        categorical=CategoricalTensor(
            code=torch.stack([wide, narrow], dim=-1).to(torch.int32),
            categories=CATEGORIES,
        ),
    )


def _context() -> TableTensor:
    wide = torch.arange(200) % 60
    wide[::7] = -1
    narrow = torch.arange(200) % 3
    return _table(wide, narrow)


def _expected(fitted: torch.Tensor, wide: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(fitted[fitted >= 0], minlength=60)
    unknown = (fitted < 0).sum()
    return torch.where(wide >= 0, counts[wide.clamp_min(0)], unknown).log1p()


def test_add_level_counts_appends_log_counts() -> None:
    context = _context()

    output = AddLevelCounts().fit_transform(context)

    assert output.columns[Stype.categorical] == ("city", "kind")
    assert output.columns[Stype.numerical] == ("city__count",)
    assert output.categorical.equal(context.categorical)
    wide = context.categorical.code[:, 0]
    torch.testing.assert_close(
        output.numerical[:, 0],
        _expected(wide, wide).to(output.numerical.dtype),
    )


def test_add_level_counts_uses_fitted_counts() -> None:
    context = _context()
    wide = torch.tensor([59, 0, -1, 7, 7, -1, 12])
    query = _table(wide, torch.tensor([0, 1, 2, -1, 0, 1, 2]))

    output = AddLevelCounts().fit(context).transform(query)

    assert output.columns[Stype.numerical] == ("city__count",)
    torch.testing.assert_close(
        output.numerical[:, 0],
        _expected(context.categorical.code[:, 0], wide).to(
            output.numerical.dtype
        ),
    )


def test_add_level_counts_keeps_low_cardinality_table() -> None:
    context = _context()

    output = AddLevelCounts(min_cardinality=100).fit_transform(context)

    assert output.equal(context)


def test_add_level_counts_fits_leading_batches_independently() -> None:
    wide = torch.stack(
        [_context().categorical.code[:, 0], torch.arange(200) % 55]
    )
    narrow = torch.stack(
        [torch.arange(200) % 3, torch.zeros(200, dtype=torch.long)]
    )
    batched = _table(wide, narrow)

    output = AddLevelCounts().fit_transform(batched)

    assert output.size() == (2, 200, 3)
    assert output.columns[Stype.numerical] == ("city__count",)
    for batch_index in range(2):
        torch.testing.assert_close(
            output.numerical[batch_index, :, 0],
            _expected(wide[batch_index], wide[batch_index]).to(
                output.numerical.dtype
            ),
        )


def test_add_level_counts_fits_ensemble_members_independently() -> None:
    first = _context()
    second = _table(torch.arange(200) % 55, torch.arange(200) % 3)
    context = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )
    processor = EnsembleProcessor.as_processor(AddLevelCounts())

    output = processor.fit_transform_ensemble(context)

    for member_id, table in enumerate((first, second)):
        member = output.table(member_id)
        assert member.columns[Stype.numerical] == ("city__count",)
        wide = table.categorical.code[:, 0]
        torch.testing.assert_close(
            member.numerical[:, 0],
            _expected(wide, wide).to(member.numerical.dtype),
        )
