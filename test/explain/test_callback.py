# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import RelatedTables, TableTensor
from sdm.explain import (
    CaptureContext,
    CaptureInputs,
    EnableContextGradients,
    EnableInputGradients,
)


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("optional", [False, True])
def test_enable_gradients(context: bool, optional: bool) -> None:
    model = torch.nn.Identity()
    original = torch.ones(2, 1, requires_grad=True)
    table = TableTensor(numerical=2 * original)
    related = (
        RelatedTables(
            tables={"related": table}, relationships=(), task_links=()
        )
        if optional
        else None
    )
    if context:
        x, y, context_only, related = (
            EnableContextGradients().on_context_preprocessing_end(
                model=model,
                x=table,
                y=table,
                x_context_only=table if optional else None,
                related_tables=related,
            )
        )
        tables = [x, y]
        if optional:
            assert context_only is not None
            tables.append(context_only)
        else:
            assert context_only is None
    else:
        x, related = EnableInputGradients().on_query_preprocessing_end(
            model=model,
            x=table,
            related_tables=related,
        )
        tables = [x]

    if optional:
        assert related is not None
        tables.append(related.tables["related"])
    else:
        assert related is None

    for result in tables:
        torch.testing.assert_close(result.numerical, table.numerical)
        result.numerical.sum().backward()
        torch.testing.assert_close(
            result.numerical.grad, torch.ones_like(original)
        )
    assert original.grad is None


@pytest.mark.parametrize("context", [False, True])
def test_capture_inputs(context: bool) -> None:
    model = torch.nn.Identity()
    x = TableTensor(numerical=torch.ones(2, 1))
    related = RelatedTables(
        tables={"related": x}, relationships=(), task_links=()
    )
    if context:
        context_capture = CaptureContext()
        for related_tables in (None, related):
            out = context_capture.on_context_preprocessing_end(
                model=model,
                x=x,
                y=x,
                x_context_only=None,
                related_tables=related_tables,
            )
            assert all(
                a is b for a, b in zip(out, (x, x, None, related_tables))
            )
        assert len(context_capture.contexts) == 2
        assert context_capture.contexts[0][-1] is None
        assert context_capture.contexts[1][-1] is related
    else:
        query_capture = CaptureInputs()
        for related_tables in (None, related):
            out_x, out_related = query_capture.on_query_preprocessing_end(
                model=model,
                x=x,
                related_tables=related_tables,
            )
            assert out_x is x
            assert out_related is related_tables
        assert len(query_capture.inputs) == 2
        assert query_capture.inputs[0][-1] is None
        assert query_capture.inputs[1][-1] is related
