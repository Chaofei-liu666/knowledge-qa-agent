"""混合检索：两阶段（文档粗排 → chunk 精排）+ RRF 融合。

第一阶段对每篇文档（离线入库时已生成摘要）打分：BM25 取该文档 chunk 最大值，
余弦取 query 与文档摘要向量的相似度，两路各自排名后 RRF 融合，再按相对阈值
截断候选文档（不固定篇数）。第二阶段只在候选文档内对 chunk 做精排。
为什么用 RRF 不用加权求和：BM25 分和 cosine 分不在一个量纲上。
"""
from __future__ import annotations

from typing import Any

from ..config import (RETRIEVAL_DOC_MAX, RETRIEVAL_DOC_RATIO,
                      RETRIEVAL_MIN_SEMANTIC_SCORE, RETRIEVAL_RRF_K,
                      RETRIEVAL_TOP_K)
from .bm25 import BM25, load_synonyms, normalize_query, tokenize
from .embed import cosine

RRF_K = RETRIEVAL_RRF_K


class Retriever:
    def __init__(self, index: dict[str, Any], vectors: dict[str, list[float]],
                 synonyms: dict[str, str] | None = None) -> None:
        self.chunks: dict[str, dict] = {c["chunk_id"]: c for c in index["chunks"]}
        self.docs: dict[str, dict] = {d["doc_id"]: d for d in index["docs"]}
        self.bm25 = BM25.from_dict(index["bm25"])
        self.vectors = vectors
        self.synonyms = synonyms if synonyms is not None else load_synonyms()
        self.embedder_kind = index.get("embedder", {}).get("kind", "hashing")

    def check_embedder(self) -> None:
        """索引与查询必须用同一个向量后端，否则余弦没有意义。"""
        from .embed import get_embedder

        current = get_embedder().kind
        if self.embedder_kind != current:
            raise RuntimeError(
                f"索引向量后端是 {self.embedder_kind}，当前是 {current}；"
                f"请用同一后端重新执行 python -m src.build_index")

    def _visible_doc(self, doc_id: str, scope: str | None, include_superseded: bool) -> bool:
        meta = self.docs.get(doc_id, {})
        if not include_superseded and meta.get("status") == "superseded":
            return False
        if scope and meta.get("scope") not in {scope, "集团", "全集团"}:
            return False
        return True

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K, scope: str | None = None,
               include_superseded: bool = True) -> list[dict]:
        normalized = normalize_query(query, self.synonyms)
        q_tokens = tokenize(normalized)

        bm25_scores = {cid: self.bm25.score(cid, q_tokens) for cid in self.chunks}
        qvec = self._embed(normalized)
        vec_scores = {cid: cosine(qvec, self.vectors.get(cid, [])) for cid in self.chunks}

        # ---- 第一阶段：文档级粗排，按相对阈值截断候选文档（不固定篇数）----
        doc_bm25: dict[str, float] = {}
        doc_vec: dict[str, float] = {}
        for cid, chunk in self.chunks.items():
            doc_id = chunk["doc_id"]
            if not self._visible_doc(doc_id, scope, include_superseded):
                continue
            doc_bm25[doc_id] = max(doc_bm25.get(doc_id, 0.0), bm25_scores[cid])
        for doc_id in doc_bm25:
            doc_vec[doc_id] = cosine(qvec, self.vectors.get(f"__doc__:{doc_id}", []))

        # 两路各自排名 → RRF 融合成文档分；某一路全为 0 时退化为单路
        docs_sorted_bm25 = sorted(doc_bm25, key=lambda d: doc_bm25[d], reverse=True)
        docs_sorted_vec = sorted(doc_vec, key=lambda d: doc_vec[d], reverse=True)
        bm25_doc_rank = {d: i + 1 for i, d in enumerate(docs_sorted_bm25)}
        vec_doc_rank = {d: i + 1 for i, d in enumerate(docs_sorted_vec)}
        doc_scores = {
            d: 1 / (RRF_K + bm25_doc_rank[d]) + 1 / (RRF_K + vec_doc_rank[d])
            for d in doc_bm25
        }
        ordered = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)
        top_doc_score = doc_scores[ordered[0]] if ordered else 0.0
        # 阈值截断：至少 1 篇，最多 DOC_MAX 篇兜底
        top_docs = [
            d for d in ordered
            if doc_scores[d] >= top_doc_score * RETRIEVAL_DOC_RATIO
        ][: RETRIEVAL_DOC_MAX] or ordered[:1]

        # ---- 第二阶段：候选文档内 chunk 精排，两路排名 RRF 融合 ----
        candidates = [cid for cid, c in self.chunks.items()
                      if c["doc_id"] in top_docs
                      and self._visible_doc(c["doc_id"], scope, include_superseded)]
        bm25_rank = {cid: i + 1 for i, cid in enumerate(
            sorted(candidates, key=lambda c: bm25_scores[c], reverse=True))}
        vec_rank = {cid: i + 1 for i, cid in enumerate(
            sorted(candidates, key=lambda c: vec_scores[c], reverse=True))}

        fused = []
        try:
            from .embed import get_embedder
            floor = getattr(get_embedder(), "min_score", RETRIEVAL_MIN_SEMANTIC_SCORE)
        except Exception:  # 后端不可用时退回配置里的默认地板
            floor = RETRIEVAL_MIN_SEMANTIC_SCORE
        for cid in candidates:
            # 相关性地板：纯关键词无命中、余弦又很低时，不把噪音当证据
            if bm25_scores[cid] <= 0 and vec_scores[cid] < floor:
                continue
            score = 1 / (RRF_K + bm25_rank[cid]) + 1 / (RRF_K + vec_rank[cid])
            fused.append((cid, score))
        fused.sort(key=lambda kv: kv[1], reverse=True)

        hits = []
        for cid, score in fused[:top_k]:
            chunk = dict(self.chunks[cid])
            chunk["score"] = round(score, 6)
            chunk["bm25"] = round(bm25_scores[cid], 4)
            chunk["cosine"] = round(vec_scores[cid], 4)
            chunk["snippet"] = chunk["text"][:200]
            hits.append(chunk)
        return hits

    def _embed(self, text: str) -> list[float]:
        self.check_embedder()
        from .embed import get_embedder
        return get_embedder().embed(text)

    # ---- 给工具层的只读视图 ----
    def get_doc(self, doc_id: str) -> dict[str, Any] | None:
        return self.docs.get(doc_id)

    def doc_chunks(self, doc_id: str) -> list[dict]:
        return [c for c in self.chunks.values() if c["doc_id"] == doc_id]

    def all_docs(self) -> list[dict]:
        return list(self.docs.values())
