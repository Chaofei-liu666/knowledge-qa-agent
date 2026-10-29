"""文档解析：front matter + 按章节切块。

不按固定长度切，是因为引用要定位到「第 5.2 条」，查证也要能直接读「适用范围」那一节。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import DOCS_DIR

HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
SECTION_ID_RE = re.compile(r"^((?:\d+\.)*\d+|[Qq]\d+|[A-Za-z]+-?\d*)\s+")


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta: dict[str, Any] = {}
    for line in fm.strip().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta, body.lstrip("\n")


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def split_sections(body: str) -> list[dict[str, str]]:
    """按 `## 标题` 切块；标题前的正文并入第 1 节。"""
    matches = list(HEADING_RE.finditer(body))
    sections: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        sid_match = SECTION_ID_RE.match(heading)
        section_id = sid_match.group(1) if sid_match else heading
        sections.append({"section_id": section_id, "heading": heading, "text": text})
    return sections


def parse_doc(path: Path) -> dict[str, Any]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    slots: dict[str, str] = {}
    for item in (meta.get("slots") or "").split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            if k.strip():
                slots[k.strip()] = v.strip()
    return {
        "doc_id": meta.get("id") or path.stem,
        "title": meta.get("title", ""),
        "version": meta.get("version", ""),
        "effective_date": meta.get("effective_date", ""),
        "scope": meta.get("scope", ""),
        "precedence": _to_int(meta.get("precedence", "")),
        "status": meta.get("status", "active"),
        "superseded_by": meta.get("superseded_by") or None,
        "topics": [t.strip() for t in (meta.get("topics") or "").split(",") if t.strip()],
        "slots": slots,
        "sections": split_sections(body),
    }


def load_docs(docs_dir: Path | None = None) -> list[dict[str, Any]]:
    docs_dir = docs_dir or DOCS_DIR
    return [parse_doc(p) for p in sorted(docs_dir.glob("*.md"))]
