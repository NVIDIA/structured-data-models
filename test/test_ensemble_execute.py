# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import operator

import pytest
import torch

from sdm import EnsembleData, EnsembleTable, EnsembleTensor, TableTensor
from sdm.processing import Standardize


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_operations_use_logical_dimensions(device: str) -> None:
    data = torch.randn(4, 7, 3, device=device)
    ensemble = EnsembleTensor(
        groups=(data,), locations=tuple((0, i) for i in range(4))
    )
    for operation in (
        lambda x: torch.mean(x, dim=0),
        lambda x: x.mean(dim=-2),
        lambda x: x.square().sum(),
        lambda x: x.reshape(-1).softmax(0),
        lambda x: torch.cat((x, x + 1), dim=1),
        lambda x: x[..., 1:],
    ):
        result = operation(ensemble)
        for member in range(4):
            torch.testing.assert_close(result[member], operation(data[member]))


def test_alignment_across_fit_and_query_groups() -> None:
    data = torch.randn(4, 5, 3)
    query = EnsembleTensor(
        groups=(data,), locations=tuple((0, i) for i in range(4))
    )
    state = EnsembleTensor(
        groups=(torch.tensor([[[1.0]], [[2.0]]]), torch.tensor([[[3.0]]])),
        locations=((0, 1), (1, 0), (0, 0), (0, 1)),
    )
    result = query - state
    for member in range(4):
        torch.testing.assert_close(
            result[member], data[member] - state[member]
        )


def test_shared_inputs_retain_shared_results() -> None:
    data = torch.randn(1, 5, 3)
    ensemble = EnsembleTensor(groups=(data,), locations=((0, 0),) * 8)
    result = ensemble.mean(0)
    assert result[0].data_ptr() == result[7].data_ptr()
    torch.testing.assert_close(result[0], data[0].mean(0))
    physical = EnsembleData.execute(torch.mean, ensemble, dim=-2, batched=True)
    torch.testing.assert_close(physical[7], data[0].mean(0))
    assert physical[0].data_ptr() == physical[7].data_ptr()
    values, indices = ensemble.sort()
    expected = data[0].sort()
    torch.testing.assert_close(values[7], expected.values)
    torch.testing.assert_close(indices[7], expected.indices)
    assert values[0].data_ptr() == values[7].data_ptr()
    assert ensemble.add_(1) is ensemble
    assert (
        EnsembleData.execute(torch.Tensor.add_, ensemble, 1, batched=True)
        is ensemble
    )
    assert torch.add(ensemble, 1, out=ensemble) is ensemble
    torch.testing.assert_close(ensemble[7], data[0])


def test_python_values_and_multiple_outputs() -> None:
    values = EnsembleData(
        groups=("first", "second"), locations=((0, 0), (1, 0), (0, 0))
    )
    result = EnsembleData.execute(operator.add, values, "!")
    assert [result[i] for i in range(3)] == ["first!", "second!", "first!"]
    tensor = EnsembleTensor(
        groups=(torch.randn(3, 4),), locations=((0, 0), (0, 1), (0, 2))
    )
    sorted_values, sorted_indices = tensor.sort()
    for member in range(3):
        expected_values, expected_indices = torch.sort(tensor[member])
        torch.testing.assert_close(sorted_values[member], expected_values)
        torch.testing.assert_close(sorted_indices[member], expected_indices)


def test_numerical_blocks_and_replacement() -> None:
    data = torch.randn(4, 5, 3)
    table = EnsembleTable(
        groups=(TableTensor.from_tensor(data),),
        locations=tuple((0, i) for i in range(4)),
    )
    result = table.replace_blocks(
        numerical=table.numerical - table.numerical.mean(-2, keepdim=True)
    )
    assert result.num_groups == 1
    for member in range(4):
        torch.testing.assert_close(
            result[member].numerical,
            data[member] - data[member].mean(-2, keepdim=True),
        )


def test_operations_reject_different_logical_member_counts() -> None:
    first = EnsembleTensor(groups=(torch.ones(1, 3),), locations=((0, 0),) * 2)
    second = EnsembleTensor(
        groups=(torch.ones(1, 3),), locations=((0, 0),) * 3
    )
    with pytest.raises(ValueError, match="same number of members"):
        first + second


def test_inplace_updates_original_storage_in_member_order() -> None:
    data = torch.zeros(3, 2)
    destination = EnsembleTensor(
        groups=(data,), locations=((0, 2), (0, 0), (0, 1))
    )
    values = EnsembleTensor(
        groups=(torch.tensor([[1.0], [2.0], [3.0]]),),
        locations=((0, 0), (0, 1), (0, 2)),
    )
    assert destination.add_(values) is destination
    torch.testing.assert_close(
        data, torch.tensor([[2.0, 2.0], [3.0, 3.0], [1.0, 1.0]])
    )
    assert torch.mul(destination, 2, out=destination) is destination
    torch.testing.assert_close(
        data, torch.tensor([[4.0, 4.0], [6.0, 6.0], [2.0, 2.0]])
    )
    destination /= 2
    torch.testing.assert_close(
        data, torch.tensor([[2.0, 2.0], [3.0, 3.0], [1.0, 1.0]])
    )


def test_shared_inplace_updates_cannot_overwrite_distinct_results() -> None:
    data = torch.zeros(1, 2)
    destination = EnsembleTensor(groups=(data,), locations=((0, 0),) * 2)
    values = EnsembleTensor(
        groups=(torch.tensor([[1.0], [2.0]]),), locations=((0, 0), (0, 1))
    )
    with pytest.raises(RuntimeError, match="shared ensemble storage"):
        destination.add_(values)
    torch.testing.assert_close(data, torch.zeros_like(data))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scalar_broadcasting_preserves_tensor_promotion(device: str) -> None:
    data = torch.randn(3, 4, 2, device=device)
    values = EnsembleTensor(groups=(data,), locations=((0, 0), (0, 1), (0, 2)))
    scalars = EnsembleTensor(
        groups=(torch.arange(3, dtype=torch.float64, device=device),),
        locations=((0, 0), (0, 1), (0, 2)),
    )
    for operand in (scalars, torch.tensor(1.0, dtype=torch.float64)):
        result = values + operand
        for member in range(3):
            expected = data[member] + (
                operand[member] if operand is scalars else operand
            )
            torch.testing.assert_close(result[member], expected)


def test_cpu_checkpoint_loads_into_cuda_processor() -> None:
    data = torch.randn(3, 8, 2)
    table = EnsembleTable(
        groups=(TableTensor.from_tensor(data),),
        locations=((0, 0), (0, 1), (0, 2)),
    )
    fitted = Standardize().fit(table)
    checkpoint = io.BytesIO()
    torch.save(fitted.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Standardize().cuda()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    result = restored.transform(table.cuda())
    expected = fitted.transform(table)
    for member in range(3):
        assert result[member].device.type == "cuda"
        torch.testing.assert_close(
            result[member].numerical.cpu(), expected[member].numerical
        )


def test_boolean_operations_do_not_mutate_their_inputs() -> None:
    data = torch.tensor([[True, False], [True, True]])
    values = EnsembleTensor(groups=(data,), locations=((0, 0), (0, 1)))
    result = values & ~values
    assert not result[0].any()
    assert not result[1].any()
    assert values[0].tolist() == [True, False]
    assert values[1].tolist() == [True, True]
