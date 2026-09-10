import torch

from sdm import TableTensor
from sdm.models.timesfm3 import TimesFM3


def test_forward() -> None:
    model = TimesFM3(pretrained=False)

    # Past-and-future covariates:
    x_context = TableTensor.from_tensor(torch.randn(5, 2))
    x_query = TableTensor.from_tensor(torch.randn(3, 2))

    # Target variates:
    y_context = TableTensor.from_tensor(
        torch.randn(5, 3),
        columns=["y1", "y2", "y3"],
    )

    out = model(x_context, y_context, x_query)
    assert out.size() == (3, 3 * 9)
    assert "y1__q10" in out.columns["numerical"]
    assert "y2__q50" in out.columns["numerical"]
    assert "y3__q90" in out.columns["numerical"]

    model.fit(x_context, y_context)
    out = model.predict(x_query)
    assert out.allclose(model.predict(x_query))
