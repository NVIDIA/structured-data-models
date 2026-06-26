import torch
from sdm.processing import Power, Quantile


def test_processor_torch_compile_smoke() -> None:
    input = torch.tensor(
        [
            [-2.0, 0.0],
            [-1.0, 10.0],
            [0.0, 20.0],
            [1.0, 30.0],
            [2.0, 40.0],
        ],
        dtype=torch.float64,
    )

    processors = [
        Power().fit(input),
        Quantile(n_quantiles=3, subsample=None).fit(input),
    ]

    for processor in processors:
        expected = processor.transform(input)
        compiled = torch.compile(
            processor.transform,
            backend="eager",
            fullgraph=True,
        )
        assert torch.equal(compiled(input), expected)
