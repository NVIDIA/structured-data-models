import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sdm import StringTensor, TableTensor
from sdm.processing.tfidf_encoder import TfidfEncoder

rng = np.random.default_rng(0)
WORDS = [
    "".join(rng.choice(list("abcdefghijklmnop"), size=rng.integers(3, 9)))
    for _ in range(500)
]


def corpus(n: int) -> list[str]:
    return [
        " ".join(rng.choice(WORDS, size=rng.integers(2, 8))) for _ in range(n)
    ]


def bench(n: int) -> None:
    texts = corpus(n)
    table = TableTensor(
        columns={"text": ("t0",)},
        text=StringTensor.from_list([[t] for t in texts]),
    )

    t0 = time.perf_counter()
    tfidf = TfidfEncoder(ngram_range=(2, 3)).fit_transform(table).numerical
    t_tfidf = time.perf_counter() - t0

    t0 = time.perf_counter()
    sk = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    sk.fit_transform(texts)
    t_sk = time.perf_counter() - t0

    print(
        f"n={n:>7}  vocab={tfidf.shape[1]:>6}  "
        f"tfidf={t_tfidf:6.2f}s  sklearn={t_sk:6.2f}s"
    )


# Accuracy on a small corpus first.
texts = corpus(200)
table = TableTensor(
    columns={"text": ("t0",)},
    text=StringTensor.from_list([[t] for t in texts]),
)
enc = TfidfEncoder(ngram_range=(2, 3))
tfidf = enc.fit_transform(table).numerical
sk = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
ref = sk.fit_transform(texts).toarray()
vocab = enc._vocabularies[0].to_pylist()
perm = [vocab.index(g) for g in sk.get_feature_names_out()]
print("accuracy: allclose =", np.allclose(tfidf[:, perm].numpy(), ref, atol=1e-6))

for n in (1_000, 10_000, 100_000):
    bench(n)
