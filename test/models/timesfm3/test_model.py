import torch

from sdm import TableTensor
from sdm.models.timesfm3 import TimesFM3


def test_forward() -> None:
    model = TimesFM3(pretrained=False)

    # Past-and-future covariates:
    x_context = TableTensor.from_tensor(torch.randn(5, 2))
    x_query = TableTensor.from_tensor(torch.randn(3, 2))

    # Target variates:
    y_context = TableTensor.from_tensor(torch.randn(5, 3))

    out = model(x_context, y_context, x_query)
    assert out.size() == (5, 3 * 9)

    model.fit(x_context, y_context)
    out = model.predict(x_query)
    assert out.allclose(model.predict(x_query))
