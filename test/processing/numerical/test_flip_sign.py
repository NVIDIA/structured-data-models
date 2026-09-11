import torch

from sdm import TableTensor
from sdm.processing import FlipSign


def test_flip_sign() -> None:
    table = TableTensor.from_tensor(torch.randn(2, 5, 3))

    processor = FlipSign()
    out = processor.fit_transform(table)
    assert processor.sign.size() == (2, 1, 3)
    assert ((processor.sign == -1) | (processor.sign == 1)).all()
    assert out.numerical.equal(table.numerical * processor.sign)
    assert processor.inverse_transform(out).equal(table)


def test_flip_sign_inverse_preserves_ordered_predictions() -> None:
    target = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    prediction = TableTensor.from_tensor(torch.tensor([[1.0, 2.0, 3.0]]))
    processor = FlipSign(probability=1.0, quantile_output=True).fit(target)

    restored = processor.inverse_transform(prediction)

    torch.testing.assert_close(
        restored.numerical,
        torch.tensor([[-3.0, -2.0, -1.0]]),
    )
