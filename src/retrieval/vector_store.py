"""FAISS 向量索引封装。

数据量到十万级才值得上 ANN：小语料用 numpy 暴力扫更准也更快。
接口刻意做成可替换的，换成 Milvus/Qdrant 时只要实现 build/search/save/load。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = None
        self.ids: list[str] = []

    def build(self, ids: list[str], matrix: np.ndarray) -> "FaissStore":
        import faiss

        assert matrix.dtype == np.float32 and matrix.shape[1] == self.dim
        index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 128
        index.add(matrix)
        self.index, self.ids = index, list(ids)
        return self

    def search(self, qvec: list[float], k: int = 10) -> list[tuple[str, float]]:
        q = np.asarray([qvec], dtype=np.float32)
        scores, idx = self.index.search(q, k)
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def save(self, index_path: Path, ids_path: Path) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        # 用 serialize 而不是 write_index：faiss 的 C++ 文件接口打不开含中文的路径
        index_path.write_bytes(bytes(faiss.serialize_index(self.index)))
        ids_path.write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, ids_path: Path, dim: int) -> "FaissStore":
        import faiss

        store = cls(dim)
        buf = np.frombuffer(index_path.read_bytes(), dtype=np.uint8)
        store.index = faiss.deserialize_index(buf)
        store.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        return store
