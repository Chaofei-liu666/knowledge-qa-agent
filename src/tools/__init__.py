"""业务工具：检索片段、读原文、列目录。

失败不抛到顶层，统一包成 ToolError，由 loop 转成一条 observation 回给模型，
因为 ReAct 的前提是错误可见、可调整。故障注入用 FAULT 环境变量，确定性可复现。
"""
from __future__ import annotations

import os
from typing import Any

from ..config import RETRIEVAL_TOP_K


class ToolError(Exception):
    def __init__(self, code: str, message: str, suggestion: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "suggestion": self.suggestion}}


def _parse_fault(spec: str) -> dict[str, int]:
    """FAULT=search:1,read:2 —— 第 1 次 search、第 2 次 read 抛错。"""
    out: dict[str, int] = {}
    for part in (spec or "").split(","):
        if ":" in part:
            name, n = part.split(":", 1)
            try:
                out[name.strip()] = int(n)
            except ValueError:
                continue
    return out


class DocsTools:
    def __init__(self, retriever: Any, fault_spec: str | None = None) -> None:
        self.retriever = retriever
        self.faults = _parse_fault(fault_spec if fault_spec is not None else os.getenv("FAULT", ""))
        self.counters: dict[str, int] = {"search": 0, "read": 0, "list": 0}
        self.queries: list[str] = []
        self.listed = False

    def _maybe_fault(self, name: str) -> None:
        self.counters[name] += 1
        if self.faults.get(name) == self.counters[name]:
            raise ToolError("index_unavailable", "检索服务暂时不可用（HTTP 503）",
                            "可改用 list_docs 确认资料范围，再用 read_doc 直接读原文")

    def search_docs(self, query: str = "", top_k: int = RETRIEVAL_TOP_K, scope: str | None = None,
                    include_superseded: bool = True) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("invalid_args", "query 不能为空", "给一个 2-20 字的问题关键词")
        if not 1 <= int(top_k) <= 20:
            raise ToolError("invalid_args", f"top_k={top_k} 越界", "top_k 取 3-8")
        self._maybe_fault("search")
        hits = self.retriever.search(query, top_k=top_k, scope=scope,
                                     include_superseded=include_superseded)
        self.queries.append(query)
        return {"query": query, "scope": scope, "total": len(hits),
                "hits": [{k: h[k] for k in ("chunk_id", "doc_id", "title", "version", "section_id",
                                            "heading", "scope", "status", "superseded_by", "precedence",
                                            "topics", "slots", "snippet", "score", "bm25", "cosine")}
                         for h in hits]}

    def read_doc(self, doc_id: str = "", section_id: str | None = None) -> dict:
        if not doc_id:
            raise ToolError("invalid_args", "doc_id 不能为空", "可先 search_docs 或用 list_docs 查可用 ID")
        self._maybe_fault("read")
        doc = self.retriever.get_doc(doc_id)
        if doc is None:
            raise ToolError("not_found", f"文档 {doc_id} 不存在", "用 list_docs 确认可用文档 ID")
        chunks = self.retriever.doc_chunks(doc_id)
        if section_id:
            chunks = [c for c in chunks if c["section_id"] == section_id]
            if not chunks:
                raise ToolError("not_found", f"{doc_id} 没有 {section_id} 这一节",
                                "不带 section_id 读全文，或换一节")
        return {
            "doc_id": doc_id, "title": doc["title"], "version": doc["version"],
            "effective_date": doc["effective_date"], "scope": doc["scope"],
            "precedence": doc["precedence"], "status": doc["status"],
            "superseded_by": doc["superseded_by"], "slots": doc["slots"],
            "sections": [{"section_id": c["section_id"], "heading": c["heading"], "text": c["text"]}
                         for c in chunks],
        }

    def list_docs(self, topic: str | None = None) -> dict:
        self._maybe_fault("list")
        docs = self.retriever.all_docs()
        if topic:
            docs = [d for d in docs if topic in " ".join(d["topics"]) or topic in d["title"]]
        self.listed = True
        return {"docs": [{k: d[k] for k in ("doc_id", "title", "version", "scope", "status",
                                            "effective_date", "topics")} for d in docs],
                "total": len(docs)}
