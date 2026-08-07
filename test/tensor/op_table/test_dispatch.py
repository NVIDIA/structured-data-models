from collections.abc import Callable

import pytest
import torch

from sdm import TableTensor

aten = torch.ops.aten


class EqualTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(  # ty: ignore[invalid-method-override]
        cls,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is aten.equal.default:
            return True
        return NotImplemented


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48.0).view(2, 3, 4, 2),
    )


def test_unrelated_subclass_gets_dispatch_opportunity() -> None:
    inp = make_table()
    other = torch.empty(0).as_subclass(EqualTensor)

    assert aten.equal.default(inp, other)
    assert aten.equal.default(other, inp)


@pytest.mark.parametrize(
    "unsupported",
    [
        pytest.param(lambda inp: aten.add.Tensor(inp, inp), id="add.Tensor"),
        pytest.param(
            lambda inp: aten.add_.Tensor(inp, inp),
            id="add_.Tensor",
        ),
        pytest.param(
            lambda inp: aten.add.out(inp, inp, out=inp),
            id="add.out",
        ),
        pytest.param(
            lambda inp: aten.copy_.default(inp, inp),
            id="copy_.default",
        ),
        pytest.param(
            lambda inp: aten.empty_like.default(inp),
            id="empty_like.default",
        ),
        pytest.param(
            lambda inp: aten.as_strided.default(
                inp,
                inp.size(),
                inp.stride(),
            ),
            id="as_strided.default",
        ),
    ],
)
def test_unsupported_boundaries(
    unsupported: Callable[[TableTensor], object],
) -> None:
    with pytest.raises(NotImplementedError, match="not supported"):
        unsupported(make_table())
