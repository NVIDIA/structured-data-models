"""Public cache behavior with a deterministic CPU encoder substitute."""

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch
from examples.kumo.relational._relarena.text import ContextPCA, QwenDocuments

import sdm
import sdm.processing as sp
from sdm.tensor import EnsembleTable


@pytest.fixture
def fake_qwen(monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            self.tokenizer = SimpleNamespace(padding_side=None)
            self.documents = []

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, documents, **kwargs):
            self.documents.extend(documents)
            return torch.tensor(
                [
                    list(hashlib.sha256(doc.encode()).digest()[:8])
                    for doc in documents
                ],
                dtype=torch.float16,
            )

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = Encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def table(values, column="title"):
    return sdm.TableTensor.from_pandas(
        df=pd.DataFrame({column: values}), stypes={column: "text"}
    )


def test_qwen_and_context_pca_remain_fp32_under_autocast(fake_qwen):
    processor = QwenDocuments(torch.device("cpu"))
    context = table(["one", "two", "three", "four", "five"])
    query = table(["six", "one"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoded = processor.transform(context)
        pca = ContextPCA(num_components=2).fit(encoded)
        result = pca.transform(processor.transform(query))
    assert encoded.numerical.dtype == torch.float32
    assert result.numerical.dtype == torch.float32
    expected = (
        ContextPCA(num_components=2)
        .fit(processor.transform(context))
        .transform(processor.transform(query))
    )
    torch.testing.assert_close(result.numerical, expected.numerical)


def test_eight_members_share_raw_query_then_use_context_pca(
    tmp_path, fake_qwen
):
    processor = QwenDocuments(
        torch.device("cpu"), cache_path=tmp_path / "vectors.sqlite"
    )
    encoder = sp.EnsembleProcessorAdapter(processor)
    recipe = sp.Recipe(features=(encoder, ContextPCA(num_components=2)))
    contexts = tuple(
        table([f"context {i} {j}" for j in range(4)]) for i in range(8)
    )
    ensemble = EnsembleTable.from_tables(
        tables=contexts, member_table_ids=range(8)
    )
    recipe.features.fit_transform_ensemble(ensemble)
    processor.encoder.documents.clear()
    query = table(["query one", "query two", "query one"])
    result = recipe.features.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )
    assert processor.encoder.documents == [
        "title=query one",
        "title=query two",
    ]
    raw = processor.transform(query)
    for index, context in enumerate(contexts):
        reference = (
            ContextPCA(num_components=2)
            .fit(processor.transform(context))
            .transform(raw)
        )
        torch.testing.assert_close(
            result.table(index).numerical, reference.numerical
        )
    processor.cache.close()
