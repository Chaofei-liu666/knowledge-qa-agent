"""离线建索引：解析 → 按章节切块 → 挂元数据 → BM25 + 向量。"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import INDEX_PATH, VECTORS_PATH
from .corpus import load_docs
from .retrieval.bm25 import BM25, load_synonyms, normalize_query, tokenize
from .retrieval.embed import get_embedder


def doc_summary(doc: dict[str, Any]) -> str:
    """离线摘要：标题 + 各节标题与首句。用于第一阶段文档级粗排。

    不用 LLM 生成，是为了建索引确定性、离线可复现；代价是摘要偏字面，
    真实场景可换成模型摘要，但字段契约不变。
    """
    parts = [doc["title"]]
    for sec in doc["sections"]:
        first = re.split(r"(?<=。)", sec["text"].strip(), maxsplit=1)[0]
        first = (first or "").strip()[:80]
        if first:
            parts.append(f"{sec['heading']}：{first}")
    return "；".join(parts)


def build_index(docs: list[dict] | None = None) -> tuple[dict[str, Any], dict[str, list[float]]]:
    docs = docs if docs is not None else load_docs()
    chunks: list[dict] = []
    for doc in docs:
        for sec in doc["sections"]:
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{sec['section_id']}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "version": doc["version"],
                "effective_date": doc["effective_date"],
                "scope": doc["scope"],
                "precedence": doc["precedence"],
                "status": doc["status"],
                "superseded_by": doc["superseded_by"],
                "topics": doc["topics"],
                "slots": doc["slots"],
                "section_id": sec["section_id"],
                "heading": sec["heading"],
                "text": sec["text"],
            })

    synonyms = load_synonyms()
    tokenized = {
        c["chunk_id"]: tokenize(normalize_query(
            " ".join([c["title"], c["heading"], " ".join(c["topics"]), c["text"]]), synonyms))
        for c in chunks
    }
    bm25 = BM25().fit(tokenized)

    embedder = get_embedder()
    vectors = {c["chunk_id"]: embedder.embed(f"{c['title']} {c['heading']} {c['text']}") for c in chunks}

    docs_out: list[dict] = []
    for doc in docs:
        meta = {k: v for k, v in doc.items() if k != "sections"}
        meta["summary"] = doc_summary(doc)
        docs_out.append(meta)
        # doc 级向量：用摘要文本嵌入，供第一阶段文档粗排
        vectors[f"__doc__:{doc['doc_id']}"] = embedder.embed(meta["summary"])

    index = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": {"kind": embedder.kind, "model": embedder.model},
        "docs": docs_out,
        "chunks": chunks,
        "bm25": bm25.to_dict(),
    }
    return index, vectors


def save_index(index: dict, vectors: dict, index_path: Path = INDEX_PATH,
               vectors_path: Path = VECTORS_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    vectors_path.write_text(json.dumps(vectors, ensure_ascii=False), encoding="utf-8")


def load_index(index_path: Path = INDEX_PATH, vectors_path: Path = VECTORS_PATH):
    index = json.loads(index_path.read_text(encoding="utf-8"))
    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    return index, vectors
