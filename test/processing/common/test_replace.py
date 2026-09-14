import torch

from sdm import Stype, TableTensor
from sdm.processing import ReplaceBlocks


def test_replace_blocks() -> None:
    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.datetime: ("date",),
        },
        numerical=torch.tensor([[2.0], [3.0]]),
        datetime=torch.tensor([[1], [2]]),
    )

    output = ReplaceBlocks(numerical=lambda block: block.square()).transform(
        table
    )

    assert output.columns == table.columns
    assert torch.equal(output.numerical, table.numerical.square())
    assert torch.equal(output.datetime, table.datetime)


def test_replace_blocks_skips_empty_blocks() -> None:
    table = TableTensor(
        columns={Stype.datetime: ("date",)},
        datetime=torch.tensor([[1], [2]]),
    )

    def fail(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("empty block callback was called")

    output = ReplaceBlocks(numerical=fail).transform(table)

    assert output is table
