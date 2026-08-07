import torch

from sdm import ColumnarTensor

aten = torch.ops.aten


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        ),
    )


def test_view_inference_state_follows_input() -> None:
    normal = make_columnar()
    with torch.inference_mode():
        normal_outputs = (
            aten.alias.default(normal),
            aten.detach.default(normal),
            aten.slice.Tensor(normal, 1, 1, 3),
        )
        inference = make_columnar()

    inference_outputs = (
        aten.alias.default(inference),
        aten.detach.default(inference),
        aten.slice.Tensor(inference, 1, 1, 3),
    )

    assert all(not out.is_inference() for out in normal_outputs)
    assert all(out.is_inference() for out in inference_outputs)
