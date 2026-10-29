"""十万级 chunk 的向量索引构建与基准：python scripts/build_scale_index.py

步骤：读合成语料（+12 篇人工语料）→ 批量嵌入 → 建 HNSW → 测延迟与召回。
产物在 data/scale/，不进仓库。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import load_docs  # noqa: E402
from src.retrieval.embed import cosine, get_embedder  # noqa: E402
from src.retrieval.vector_store import FaissStore  # noqa: E402

SCALE_DIR = ROOT / "data" / "scale"
GOLD = [
    ("采购 30 万需要谁审批？", {"PRC-001"}),
    ("报销期限多久？", {"FIN-001", "FIN-004"}),
    ("密码多久换一次？", {"IT-001", "IT-002"}),
    ("年假有几天？", {"HR-001"}),
    ("退款政策是什么？", {"PRD-001"}),
    ("试用期多久？", {"HR-002"}),
]


def load_chunks() -> list[dict]:
    path = SCALE_DIR / "chunks.jsonl"
    if not path.exists():
        raise SystemExit("缺少 data/scale/chunks.jsonl，先运行 scripts/gen_scale_corpus.py")
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    # 把 12 篇人工语料也放进去，作为"针"
    for doc in load_docs():
        for sec in doc["sections"]:
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{sec['section_id']}", "doc_id": doc["doc_id"],
                "title": doc["title"], "version": doc["version"],
                "effective_date": doc["effective_date"], "scope": doc["scope"],
                "precedence": doc["precedence"], "status": doc["status"],
                "superseded_by": doc["superseded_by"], "topics": doc["topics"],
                "slots": doc["slots"], "section_id": sec["section_id"],
                "heading": sec["heading"], "text": sec["text"],
            })
    return chunks


def main() -> int:
    embedder = get_embedder()
    chunks = load_chunks()
    print(f"chunk 总数：{len(chunks)}")

    t0 = time.time()
    texts = [f"{c['title']} {c['heading']} {c['text']}" for c in chunks]
    cache = SCALE_DIR / f"matrix-{len(chunks)}-{embedder.model.replace('/', '_')}.npy"
    if cache.exists():
        matrix = np.load(cache)
        print(f"复用向量缓存 {cache.name}")
    else:
        matrix = embedder.embed_batch(texts).astype(np.float32)
        np.save(cache, matrix)
    t_embed = time.time() - t0
    print(f"嵌入耗时：{t_embed:.1f}s，矩阵 {matrix.shape}，{matrix.nbytes / 1e6:.0f}MB")

    t0 = time.time()
    store = FaissStore(dim=matrix.shape[1]).build([c["chunk_id"] for c in chunks], matrix)
    t_build = time.time() - t0
    store.save(SCALE_DIR / "faiss.index", SCALE_DIR / "ids.json")
    size_mb = (SCALE_DIR / "faiss.index").stat().st_size / 1e6
    print(f"建索引：{t_build:.1f}s，索引文件 {size_mb:.0f}MB")

    # 延迟：ANN vs 暴力
    queries = [q for q, _ in GOLD] * 20
    qvecs = [embedder.embed(q) for q in queries]

    t0 = time.time()
    for v in qvecs:
        store.search(v, k=5)
    ann_ms = (time.time() - t0) / len(qvecs) * 1000

    qmat = np.asarray(qvecs, dtype=np.float32)
    t0 = time.time()
    for v in qmat:
        scores = matrix @ v
        np.argpartition(scores, -5)[-5:]
    brute_ms = (time.time() - t0) / len(qmat) * 1000

    # 召回：人工语料的 6 个问题，看金标文档是否进前 5
    pos = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    hits = 0
    ranks = []
    for q, gold_docs in GOLD:
        found = store.search(embedder.embed(q), k=5)
        docs = {chunks[pos[cid]]["doc_id"] for cid, _ in found if cid in pos}
        hit = bool(docs & gold_docs)
        hits += int(hit)
        ranks.append({"query": q, "hit@5": hit, "top5": sorted(docs)[:5]})

    result = {
        "chunks": len(chunks),
        "embedder": f"{embedder.kind}:{embedder.model}",
        "dim": int(matrix.shape[1]),
        "embed_seconds": round(t_embed, 1),
        "embed_tokens": getattr(embedder, "tokens", None),
        "build_seconds": round(t_build, 1),
        "index_mb": round(size_mb, 1),
        "ann_ms_per_query": round(ann_ms, 2),
        "brute_ms_per_query": round(brute_ms, 2),
        "recall_at_5": f"{hits}/{len(GOLD)}",
        "per_query": ranks,
    }
    (SCALE_DIR / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
