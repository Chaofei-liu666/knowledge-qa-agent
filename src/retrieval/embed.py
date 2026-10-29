"""语义向量。

两种实现：
- ApiEmbedder：真正的语义向量，走 OpenAI 兼容 /embeddings 接口（需配置 EMBED_BASE_URL / EMBED_API_KEY）。
- HashingEmbedder：离线兜底，字符 n-gram 哈希到固定维度。它能跑通链路，但不是真语义检索，
  只用于没有 embedding 服务时的演示与测试，README 与 03-evaluation 里都标注了这条边界。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request

from ..config import EMBED_API_KEY, EMBED_BASE_URL, EMBED_DIM, EMBED_MODEL


class HashingEmbedder:
    kind = "hashing"
    # 相关性地板：不同向量后端的分数量纲不一样，阈值必须跟着后端走
    min_score = 0.30

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.model = f"hashing-{dim}"

    def embed_batch(self, texts: list[str]):
        """批量嵌入，返回 float32 矩阵。numpy 只在规模构建时用到。"""
        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self.embed(text)
        return out

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        clean = "".join(ch for ch in text if ch.strip())
        for n in (2, 3):
            for i in range(max(0, len(clean) - n + 1)):
                gram = clean[i : i + n]
                h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h, "big") % self.dim
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class ApiEmbedder:
    kind = "api"
    # 真语义的余弦整体偏高：无关问题常见 0.3-0.4，相关问题 0.6 以上
    min_score = float(os.getenv("EMBED_MIN_SCORE_API", "0.45"))

    def __init__(self) -> None:
        self.model = EMBED_MODEL
        self.batch_size = int(os.getenv("EMBED_BATCH_SIZE", "10"))
        self.workers = int(os.getenv("EMBED_WORKERS", "8"))
        self.tokens = 0

    def _request(self, batch: list[str], attempt: int = 0) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            f"{EMBED_BASE_URL.rstrip('/')}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {EMBED_API_KEY}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "ignore")[:300]
            transient = exc.code in (429, 500, 502, 503) or (exc.code == 400 and "Arrearage" in body)
            if transient and attempt < 6:
                time.sleep(3.0 * (attempt + 1))
                return self._request(batch, attempt + 1)
            raise RuntimeError(f"embedding 接口返回 {exc.code}：{body}") from exc
        self.tokens += int(data.get("usage", {}).get("prompt_tokens", 0))
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]

    def embed(self, text: str) -> list[float]:
        vec = self._request([text])[0]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]):
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor

        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        vectors: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for out in pool.map(self._request, batches):
                vectors.extend(out)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class FastEmbedEmbedder:
    """本地语义向量（BAAI/bge-small-zh-v1.5，ONNX，CPU 可跑）。

    首次使用会下载约 100MB 模型；适合离线演示真语义，代价是多两个依赖。
    """

    kind = "fastembed"
    min_score = float(os.getenv("EMBED_MIN_SCORE_LOCAL", "0.55"))

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.model = f"fastembed-{model_name}"
        self._encoder = TextEmbedding(model_name)
        self.dim = len(self.embed("维度探测"))

    def embed(self, text: str) -> list[float]:
        return next(iter(self._encoder.embed([text]))).tolist()

    def embed_batch(self, texts: list[str]):
        import numpy as np

        vecs = list(self._encoder.embed(texts, batch_size=128))
        return np.asarray(vecs, dtype=np.float32)


def get_embedder() -> HashingEmbedder | ApiEmbedder | FastEmbedEmbedder:
    provider = os.getenv("EMBED_PROVIDER", "").strip().lower()
    if provider == "fastembed":
        return FastEmbedEmbedder(os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"))
    if provider == "api" and EMBED_BASE_URL and EMBED_API_KEY:
        return ApiEmbedder()
    if not provider and EMBED_BASE_URL and EMBED_API_KEY:
        return ApiEmbedder()
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    # 两边都已归一化，直接点积
    return sum(x * y for x, y in zip(a, b))
