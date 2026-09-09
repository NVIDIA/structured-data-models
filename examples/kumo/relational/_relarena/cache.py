"""Bounded persistent memoization of frozen raw document vectors."""

import hashlib
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch


class DocumentCache:
    """Store FP32 vectors before PCA with a cap on vector payload bytes."""

    def __init__(
        self,
        path: Path,
        *,
        identity: str,
        dimension: int,
        max_vector_bytes: int,
        device: torch.device,
    ) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-8192")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata "
            "(identity TEXT, dimension INTEGER)"
        )
        saved = self.connection.execute(
            "SELECT identity, dimension FROM metadata"
        ).fetchone()
        if saved is not None and saved != (identity, dimension):
            self.connection.close()
            raise ValueError("Cache encoder configuration does not match")
        if saved is None:
            self.connection.execute(
                "INSERT INTO metadata VALUES (?, ?)", (identity, dimension)
            )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS vectors "
            "(key BLOB PRIMARY KEY, value BLOB) WITHOUT ROWID"
        )
        self.connection.commit()
        self.dimension = dimension
        self.device = device
        self.capacity = max_vector_bytes // (dimension * 4)
        self.rows = self.connection.execute(
            "SELECT COUNT(*) FROM vectors"
        ).fetchone()[0]
        if self.rows > self.capacity:
            self.connection.close()
            raise ValueError("Existing cache exceeds requested payload budget")

    def encode(
        self,
        documents: Sequence[str],
        encoder: Callable[[Sequence[str]], torch.Tensor],
    ) -> torch.Tensor:
        # Preserve duplicate/order semantics even without caller deduplication.
        unique = list(dict.fromkeys(documents))
        positions = {document: index for index, document in enumerate(unique)}
        keys = [
            hashlib.sha256(document.encode("utf8")).digest()
            for document in unique
        ]
        vectors = np.empty((len(unique), self.dimension), dtype=np.float32)
        missing = []
        for index, key in enumerate(keys):
            row = self.connection.execute(
                "SELECT value FROM vectors WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                missing.append(index)
            else:
                vectors[index] = np.frombuffer(row[0], dtype=np.float32)
        if missing:
            # Let the encoder length-sort all misses before internal batching.
            with torch.inference_mode():
                encoded = (
                    encoder([unique[index] for index in missing])
                    .float()
                    .cpu()
                    .numpy()
                )
            for index, vector in zip(missing, encoded, strict=True):
                vectors[index] = vector
                if self.rows < self.capacity:
                    self.connection.execute(
                        "INSERT INTO vectors VALUES (?, ?)",
                        (keys[index], vector.tobytes()),
                    )
                    self.rows += 1
        self.connection.commit()
        order = [positions[document] for document in documents]
        return torch.from_numpy(vectors[order]).to(self.device)

    def close(self) -> None:
        self.connection.close()
