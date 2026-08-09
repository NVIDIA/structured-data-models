from __future__ import annotations

import pytest
import torch

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import SentenceTransformer
from sdm.testing import withCUDA


@withCUDA
def test_forward(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentence_transformers = pytest.importorskip("sentence_transformers")

    class _SentenceTransformer:
        values: list[str]

        def __init__(self, model_name: str) -> None:
            assert model_name == "dummy"

        def get_embedding_dimension(self) -> int:
            return 4

        def encode(
            self,
            values: list[str],
            *,
            show_progress_bar: bool,
            convert_to_tensor: bool,
            device: str,
            batch_size: int,
        ) -> torch.Tensor:
            assert not show_progress_bar
            assert convert_to_tensor
            assert batch_size == 16
            data = [{"": 0, "a": 1, "b": 2, "c": 3}[value] for value in values]
            return (
                torch.tensor(data, device=device).unsqueeze(-1).expand(-1, 4)
            )

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        _SentenceTransformer,
    )

    table = TableTensor(
        columns={"text": ("title", "body")},
        text=StringTensor.from_list(
            [
                ["a", "b"],
                ["c", None],
            ],
            device=device,
        ),
    )

    processor = SentenceTransformer("dummy", batch_size=16).to(device)
    output = processor(table)

    assert output.columns[Stype.numerical] == tuple(
        f"{column}__emb{i}" for column in ("title", "body") for i in range(4)
    )
    assert output.numerical.shape == (2, 8)
    assert output.numerical.device == device
    assert output.numerical.equal(
        torch.tensor(
            [
                [1, 1, 1, 1, 2, 2, 2, 2],
                [3, 3, 3, 3, 0, 0, 0, 0],
            ],
            dtype=output.numerical.dtype,
            device=device,
        )
    )
