import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from sdm import StringTensor, TableTensor
from sdm.processing.pca import PCA
from sdm.processing.slice_features import SliceFeatures
from sdm.processing.text.llm_encoder import LLMEncoder
from sklearn.decomposition import PCA as SklearnPCA

rng = np.random.default_rng(0)
WORDS = [
    "".join(rng.choice(list("abcdefghijklmnop"), size=rng.integers(3, 9)))
    for _ in range(500)
]


def corpus(n: int) -> list[str]:
    return [
        " ".join(rng.choice(WORDS, size=rng.integers(2, 8))) for _ in range(n)
    ]


def _table(texts: list[str], device: str) -> TableTensor:
    return TableTensor(
        columns={"text": ("text",)},
        text=StringTensor.from_list(
            [[text] for text in texts],
            device=device,
        ),
    )


def _sync(device: str) -> None:
    # Async CUDA work must be flushed before reading the wall clock.
    if device == "cuda":
        torch.cuda.synchronize()


class LengthFourierEmbedder:
    """Cheap deterministic embedder for benchmarking processor overhead.

    The embedding is a dense feature map of UTF-8 string lengths. This is not
    intended to model embedding quality; it keeps the embedding work stable so
    PCA and slicing costs can be compared across SDM and sklearn pipelines.
    """

    def __init__(self, dim: int, *, dtype: torch.dtype = torch.float32):
        self._dim = dim
        self._dtype = dtype
        self._freq = torch.linspace(0.01, 1.0, dim, dtype=dtype)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, strings: "pa.Array | object") -> torch.Tensor:
        if isinstance(strings, pa.Array):
            lengths = pc.call_function("utf8_length", [strings])
            values = torch.from_numpy(
                lengths.to_numpy(zero_copy_only=False)
            ).to(self._dtype)
            return self._features(values)

        cudf_values = strings.str.len()
        values = torch.from_dlpack(cudf_values.astype("float32").to_dlpack())
        values = values.to(dtype=self._dtype)
        return self._features(values)

    def encode_numpy(self, texts: list[str]) -> np.ndarray:
        lengths = np.fromiter((len(text) for text in texts), dtype=np.float32)
        freq = self._freq.cpu().numpy()
        return np.sin(lengths[:, None] * freq[None, :]).astype(np.float32)

    def _features(self, values: torch.Tensor) -> torch.Tensor:
        freq = self._freq.to(device=values.device)
        return (values[:, None] * freq[None, :]).sin()


def bench_sdm(
    texts: list[str],
    *,
    device: str,
    embed_dim: int,
    reduce_dim: int,
    method: str,
) -> tuple[float, float, float, tuple[int, int]]:
    table = _table(texts, device)
    embedder = LengthFourierEmbedder(embed_dim)
    encoder = LLMEncoder(embedder, dtype=torch.float32)
    reducer = (
        PCA(dim=reduce_dim)
        if method == "pca"
        else SliceFeatures(dim=reduce_dim)
    )

    _sync(device)
    start = time.perf_counter()
    encoded = encoder.transform(table)
    _sync(device)
    encode_s = time.perf_counter() - start

    start = time.perf_counter()
    reduced = reducer.fit_transform(encoded)
    _sync(device)
    reduce_s = time.perf_counter() - start

    return (
        encode_s,
        reduce_s,
        encode_s + reduce_s,
        tuple(reduced.numerical.shape),
    )


def bench_sklearn(
    texts: list[str],
    *,
    embed_dim: int,
    reduce_dim: int,
    method: str,
) -> tuple[float, float, float, tuple[int, int]]:
    embedder = LengthFourierEmbedder(embed_dim)

    start = time.perf_counter()
    encoded = embedder.encode_numpy(texts)
    encode_s = time.perf_counter() - start

    start = time.perf_counter()
    if method == "pca":
        reduced = SklearnPCA(n_components=reduce_dim).fit_transform(encoded)
    else:
        reduced = encoded[:, :reduce_dim]
    reduce_s = time.perf_counter() - start

    return encode_s, reduce_s, encode_s + reduce_s, tuple(reduced.shape)


def bench(n: int, *, embed_dim: int, reduce_dim: int, method: str) -> None:
    texts = corpus(n)
    sk_encode, sk_reduce, sk_total, sk_shape = bench_sklearn(
        texts,
        embed_dim=embed_dim,
        reduce_dim=reduce_dim,
        method=method,
    )
    print(
        f"backend=sklearn method={method:>5} n={n:>7} "
        f"shape={sk_shape!s:>13} encode={sk_encode:6.2f}s "
        f"reduce={sk_reduce:6.2f}s total={sk_total:6.2f}s"
    )

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        sdm_encode, sdm_reduce, sdm_total, sdm_shape = bench_sdm(
            texts,
            device=device,
            embed_dim=embed_dim,
            reduce_dim=reduce_dim,
            method=method,
        )
        print(
            f"backend=sdm     method={method:>5} device={device:>4} "
            f"n={n:>7} shape={sdm_shape!s:>13} "
            f"encode={sdm_encode:6.2f}s reduce={sdm_reduce:6.2f}s "
            f"total={sdm_total:6.2f}s"
        )


for method in ("slice", "pca"):
    for n in (1_000, 10_000, 100_000):
        bench(n, embed_dim=768, reduce_dim=64, method=method)
