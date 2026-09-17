# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the error-correcting output code processors."""

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

import sdm
import sdm.processing as sp
from sdm.testing import onlyCUDA

NUM_CLASSES = 12
NUM_MEMBERS = 8
# Deliberately scrambled, so a decoder that returns internal category order
# instead of the original labels cannot pass.
LABELS = np.tile(np.array([7, 2, 11, 0, 5, 1, 10, 4, 9, 3, 8, 6]), 3)


def _fit_target(
    num_members: int,
) -> tuple[sp.EncodeECOC, sdm.EnsembleTable]:
    target = sdm.TableTensor.from_pandas(
        df=pd.DataFrame({"y": LABELS}),
        stypes={"y": "categorical"},
    )
    ecoc = sp.EncodeECOC(alphabet_size=10)
    pipeline = sp.EnsembleProcessor.as_processor(
        [sp.AlignCategories(), ecoc, sp.ShuffleCategories(method="shift")]
    )
    ensemble = sdm.EnsembleTable.from_table(target, num_members=num_members)
    transformed = pipeline.fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(0),
    )
    return ecoc, transformed


def _oracle(table: sdm.TableTensor, symbols: Sequence[int]) -> Tensor:
    """Return logits over ``symbols`` that name the member's own symbol."""
    code = table.categorical.code.squeeze(-1).to(torch.long)
    told = table.categorical.categories[0][code]
    position = {symbol: index for index, symbol in enumerate(symbols)}
    index = torch.tensor([position[int(symbol)] for symbol in told.tolist()])
    logits = torch.full((len(LABELS), len(symbols)), -10.0)
    return logits.scatter_(1, index[:, None], 10.0)


def _member_logits(
    transformed: sdm.EnsembleTable,
) -> list[sdm.TableTensor]:
    tables = []
    for member_id in range(transformed.num_members):
        table = transformed.table(member_id)
        order = [
            int(symbol) for symbol in table.categorical.categories[0].tolist()
        ]
        tables.append(
            sdm.TableTensor(
                columns={sdm.Stype.numerical: [str(s) for s in order]},
                numerical=_oracle(table, order),
            )
        )
    return tables


def _assert_recovers_the_labels(reduced: sdm.TableTensor) -> None:
    final = sp.Softmax(temperature=1.0).transform(reduced)
    names = final.columns[sdm.Stype.numerical]
    assert sorted(int(name) for name in names) == list(range(NUM_CLASSES))
    predicted = np.array(
        [int(names[index]) for index in final.numerical.argmax(dim=-1)]
    )
    np.testing.assert_array_equal(predicted, LABELS)


# The second count exceeds the code count, so the codebook repeats and several
# members share one code. That is the nested arrangement.
@pytest.mark.parametrize("num_members", [NUM_MEMBERS, 2 * NUM_MEMBERS])
def test_ecoc_round_trip_recovers_the_original_labels(
    num_members: int,
) -> None:
    """An oracle model on every member must decode back to the true class."""
    ecoc, transformed = _fit_target(num_members)
    assert ecoc.active
    assert ecoc.codebook.size(0) == num_members

    tables = _member_logits(transformed)

    decoder = sp.DecodeECOC(ecoc)
    reducer = sp.ReduceEstimators(method="mean")
    # The recipe stacks the member outputs and the contract drives them one by
    # one. Both must recover the labels.
    stacked = cast(sdm.TableTensor, torch.stack(tuple(tables)))
    _assert_recovers_the_labels(reducer.transform(decoder.transform(stacked)))
    _assert_recovers_the_labels(
        reducer.transform_ensemble(
            decoder.transform_ensemble(
                sdm.EnsembleTable.from_tables(
                    tables=tables,
                    member_table_ids=range(num_members),
                )
            )
        ).table(0)
    )


def test_encoder_rejects_members_that_disagree_on_the_categories() -> None:
    """One codebook maps every member, so the classes must line up."""
    target = sdm.TableTensor.from_pandas(
        df=pd.DataFrame({"y": LABELS}),
        stypes={"y": "categorical"},
    )
    ensemble = sdm.EnsembleTable.from_table(target, num_members=NUM_MEMBERS)
    # Relabels each member on its own, which is what the encoder cannot take.
    shuffled = sp.ShuffleCategories(method="shift").fit_transform_ensemble(
        ensemble,
        generator=torch.Generator().manual_seed(0),
    )

    with pytest.raises(ValueError, match="same target categories"):
        sp.EncodeECOC(alphabet_size=10).fit_ensemble(shuffled)


@onlyCUDA
def test_decoder_reads_a_codebook_from_another_device() -> None:
    """The encoder holds the codebook, so it may lag behind the scores."""
    ecoc, transformed = _fit_target(NUM_MEMBERS)
    tables = _member_logits(transformed)
    stacked = cast(sdm.TableTensor, torch.stack(tuple(tables)))
    scores = cast(sdm.TableTensor, stacked.to(torch.device("cuda")))

    decoded = sp.DecodeECOC(ecoc).transform(scores)

    assert decoded.device.type == "cuda"
    reduced = sp.ReduceEstimators(method="mean").transform(decoded)
    _assert_recovers_the_labels(cast(sdm.TableTensor, reduced.cpu()))
