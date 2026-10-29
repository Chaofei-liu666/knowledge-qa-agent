"""Agent Loop：状态、决策协议、证据信号、两道闸门、trace。

程序侧不选动作，只做三件事：
1. 执行模型选的动作，把结果和信号回灌上下文；
2. 在 answer 之前做校验（引用逐字、冲突已解释、不超引用范围、不靠废止文档）；
3. 记录每一步，供复现与审计。
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import (MAX_STEPS, MAX_SUB_QUESTIONS, MAX_TOOL_CALLS, SCOPE_KEYWORDS,
                      STATUS_ZH)
from ..llm import LLMError
from ..tools import DocsTools, ToolError

ACTION_ENUM = {"search", "read", "list", "answer", "ask", "abstain"}
ACTION_ALIASES = {
    "search_docs": "search", "retrieve": "search", "query": "search",
    "read_doc": "read", "open": "read",
    "list_docs": "list", "catalog": "list",
    "clarify": "ask", "ask_user": "ask",
    "abstain_info": "abstain", "decline": "abstain", "insufficient": "abstain",
}

# 只挑「带单位的数量」做引用覆盖检查：条款号、版本号不是结论，不该被当成数字断言
NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-])(\d+(?:\.\d+)?)\s*"
    r"(个自然日|自然日|个工作日|工作日|个月|万元|亿元|千元|万|亿|千|元|%|天|日|年|次|倍|小时)")
PUNCT_RE = re.compile(r"[\s，。；：、\"'“”「」『』（）()\[\]【】,.;:!?！？]")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return PUNCT_RE.sub("", text).lower()


def numbers(text: str) -> set[float]:
    return {float(m.group(1)) for m in NUM_RE.finditer(text or "")}


def detect_scope(text: str) -> str | None:
    for scope, kws in SCOPE_KEYWORDS.items():
        if any(k in text for k in kws):
            return scope
    return None


@dataclass
class Result:
    question: str
    status: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    note: str = ""

    @property
    def status_zh(self) -> str:
        return STATUS_ZH.get(self.status, self.status)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status_zh"] = self.status_zh
        return data


class Guard:
    """证据信号 + 校验。只拒绝，不替模型写答案。"""

    # ---------- 信号 ----------
    def scan(self, state: dict) -> dict:
        docs: dict[str, dict] = {}
        for h in state["hits"]:
            docs.setdefault(h["doc_id"], {
                "meta": h, "sections": {h["section_id"]: h["snippet"]},
            })
        for doc_id, doc in state["reads"].items():
            docs[doc_id] = {
                "meta": {**doc, "section_id": doc["sections"][0]["section_id"] if doc["sections"] else ""},
                "sections": {s["section_id"]: s["text"] for s in doc["sections"]},
            }

        slots: dict[str, list[dict]] = {}
        for doc_id, item in docs.items():
            meta = item["meta"]
            if meta.get("status") != "active":
                continue
            for slot, value in (meta.get("slots") or {}).items():
                section_id, quote = self._find_value(item["sections"], value)
                slots.setdefault(slot, []).append({
                    "doc_id": doc_id, "value": value, "precedence": meta.get("precedence"),
                    "scope": meta.get("scope"), "section_id": section_id, "quote": quote,
                })

        conflicts = []
        for slot, values in slots.items():
            if len({v["value"] for v in values}) < 2:
                continue
            # 同一取值多份文件时，保留"最弱证据"（无优先级那份），因为它才是冲突的根源
            dedup: dict[str, dict] = {}
            for v in values:
                cur = dedup.get(v["value"])
                if cur is None or (cur["precedence"] is not None and v["precedence"] is None):
                    dedup[v["value"]] = v
            vals = list(dedup.values())
            precedences = [v["precedence"] for v in vals]
            unresolved = any(p is None for p in precedences) or len(set(precedences)) == 1
            winner = None if unresolved else min(vals, key=lambda v: v["precedence"])["doc_id"]
            conflicts.append({"slot": slot, "values": vals, "unresolved": unresolved,
                              "winner": winner})

        versions = []
        for doc_id, item in docs.items():
            if item["meta"].get("status") == "superseded":
                versions.append({"doc_id": doc_id,
                                 "superseded_by": item["meta"].get("superseded_by") or ""})

        scopes: list[str] = []
        for values in slots.values():
            scopes += [v["scope"] for v in values if v["scope"]]
        ambiguous = not state["constraints"].get("scope") and len(set(scopes)) > 1

        return {"conflicts": conflicts, "versions": versions,
                "scope": {"ambiguous": ambiguous, "values": sorted(set(scopes))},
                "empty": not docs}

    @staticmethod
    def _find_value(sections: dict[str, str], value: str) -> tuple[str, str]:
        """定位槽位取值所在的小节与原句，供冲突追问引用。"""
        match = NUM_RE.search(value)
        num = match.group(1) if match else ""
        unit = (match.group(2) if match else value)[:2]
        for section_id, text in sections.items():
            for sentence in re.split(r"(?<=。)", text):
                if (not num or num in sentence) and (not unit or unit in sentence):
                    return section_id, sentence.strip()
        for section_id, text in sections.items():
            if unit and unit in text:
                return section_id, text[:120]
        first_id, first_text = next(iter(sections.items()), ("", ""))
        return first_id, (first_text or "")[:120]

    # ---------- 校验 ----------
    def verify_answer(self, args: dict, state: dict) -> list[str]:
        violations: list[str] = []
        citations = args.get("citations") or []
        if not citations:
            return ["缺少引用：每个结论都要挂原文"]

        evidence_docs = {h["doc_id"] for h in state["hits"]} | set(state["reads"])
        quotes: list[str] = []
        superseded_only = True
        for c in citations:
            doc_id = c.get("doc_id", "")
            if doc_id not in evidence_docs:
                violations.append(f"引用 {doc_id} 不在本轮证据中")
                continue
            meta = self._meta(state, doc_id)
            if meta.get("status") == "active":
                superseded_only = False
            if not c.get("quote"):
                continue
            text = self._section_text(state, doc_id, c.get("section_id", ""))
            if norm(c["quote"]) not in norm(text):
                violations.append(f"引用不是原文逐字：{doc_id} {c.get('section_id','')}")
            quotes.append(c["quote"])
        if superseded_only:
            violations.append("只引用了已废止文档")

        sig = state["signals"]
        cited_docs = {c.get("doc_id") for c in citations}
        for conflict in sig.get("conflicts", []):
            if not conflict["unresolved"]:
                if conflict.get("winner") and conflict["winner"] not in cited_docs:
                    violations.append(f"冲突有优先级，必须引用优先文件 {conflict['winner']}")
                continue
            need_docs = {v["doc_id"] for v in conflict["values"]}
            # 结构化校验：citations 里必须同时出现双方 doc_id；正文如何展开由模型自由发挥
            if not need_docs <= cited_docs:
                violations.append(f"冲突未解释：{conflict['slot']} 需要同时引用双方文档 {sorted(need_docs)}")

        question_nums = numbers(state["normalized"])
        quote_nums: set[float] = set()
        for q in quotes:
            quote_nums |= numbers(q)
        extra = numbers(args.get("answer", "")) - question_nums - quote_nums
        if extra:
            violations.append(f"结论超出引用范围：{sorted(extra)} 未出现在任何引用中")
        return violations

    def verify_ask(self, args: dict, state: dict) -> list[str]:
        if not args.get("question"):
            return ["追问缺少问题"]
        conflict = next((c for c in state["signals"].get("conflicts", []) if c["unresolved"]), None)
        if conflict:
            cited = {c.get("doc_id") for c in (args.get("citations") or [])}
            need = {v["doc_id"] for v in conflict["values"]}
            if not need <= cited:
                return [f"冲突追问必须引用双方：需要 {sorted(need)}"]
            return []
        if args.get("reason_type") not in {"scope", "missing_entity", "conflict"}:
            return ["追问缺少原因类型（scope/missing_entity/conflict）"]
        return []

    def verify_abstain(self, args: dict, state: dict) -> list[str]:
        violations = []
        if len(set(state["queries"])) < 2:
            violations.append("只检索了一次就说没有，至少换两组说法再确认")
        if not state["listed"]:
            violations.append("没有列目录确认语料范围")
        if not args.get("gaps"):
            violations.append("缺资料不足的说明（gaps）")
        return violations

    @staticmethod
    def _meta(state: dict, doc_id: str) -> dict:
        if doc_id in state["reads"]:
            return state["reads"][doc_id]
        for h in state["hits"]:
            if h["doc_id"] == doc_id:
                return h
        return {}

    @staticmethod
    def _section_text(state: dict, doc_id: str, section_id: str) -> str:
        if doc_id in state["reads"]:
            for s in state["reads"][doc_id]["sections"]:
                if s["section_id"] == section_id:
                    return s["text"]
            return "\n".join(s["text"] for s in state["reads"][doc_id]["sections"])
        for h in state["hits"]:
            if h["doc_id"] == doc_id and h["section_id"] == section_id:
                return h["snippet"]
        return ""


class Agent:
    def __init__(self, tools: DocsTools, policy: Any, trace_path: str | Path | None = None) -> None:
        self.tools = tools
        self.policy = policy
        self.guard = Guard()
        self.trace_path = Path(trace_path) if trace_path else None
        self.events: list[dict] = []

    # ---------- 入口 ----------
    def run(self, question: str, history: list[dict] | None = None) -> Result:
        state = self._init_state(question, history or [])
        self.events = []
        answer_attempts = 0

        while True:
            if state["step"] >= MAX_STEPS or state["tool_calls"] >= MAX_TOOL_CALLS:
                return self._finish(state, "incomplete",
                                    note="达到步数或工具调用上限，任务未完成",
                                    gaps=["预算耗尽，未得出结论"])

            state["step"] += 1
            state["evidence"] = self._build_evidence(state)
            try:
                decision = self.policy.decide(state)
            except LLMError as exc:
                return self._finish(state, "failed", note=f"模型调用失败：{exc}")

            action = str(decision.get("action", ""))
            action = ACTION_ALIASES.get(action, action)
            args = decision.get("args")
            if not isinstance(args, dict) or not args:
                # 模型有时把参数平铺在顶层，归一化一下，别当作空参数
                args = {k: v for k, v in decision.items() if k not in ("action", "args", "reason")}
            reason = str(decision.get("reason", ""))[:60]

            # 同一个动作、同样的参数再来一次：结果不会变，明确提醒换动作
            key = f"{action}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
            state["repeats"] = state.get("repeats", 0) + 1 if key == state.get("last_key") else 0
            state["last_key"] = key
            state["seen"][key] = state["seen"].get(key, 0) + 1

            if action not in ACTION_ENUM:
                state["violations"] = [f"未知动作 {action}"]
                self._log(state, action, args, reason, "非法动作")
                continue

            if action in {"search", "read", "list"}:
                if state["seen"][key] > 2:
                    # 同样的调用第 3 次出现：跳过执行，把"没新信息"作为观察回灌
                    state["repeats"] = state["seen"][key]
                    state["last_observation"] = "重复调用已跳过：与之前完全相同的参数，结果不会变"
                    self._log(state, action, args, reason, "重复调用已跳过")
                    continue
                if state["tool_calls"] >= MAX_TOOL_CALLS:
                    return self._finish(state, "incomplete", note="工具调用次数到顶", gaps=["预算耗尽"])
                state["tool_calls"] += 1
                digest = self._exec(action, args, state)
                state["signals"] = self.guard.scan(state)
                self._log(state, action, args, reason, digest)
                continue

            if action == "answer":
                violations = self.guard.verify_answer(args, state)
                if not violations:
                    self._log(state, action, args, reason, "校验通过")
                    return self._finish(state, "answered",
                                        answer=args.get("answer", ""),
                                        citations=args.get("citations") or [])
                answer_attempts += 1
                state["violations"] = violations
                self._log(state, action, args, reason, "校验拒绝：" + "；".join(violations))
                if answer_attempts >= 2:
                    return self._finish(state, "incomplete", note="回答两次未通过校验",
                                        gaps=violations)
                continue

            if action == "ask":
                violations = self.guard.verify_ask(args, state)
                if not violations:
                    self._log(state, action, args, reason, "追问校验通过")
                    return self._finish(state, "needs_clarification",
                                        answer=args.get("question", ""),
                                        citations=args.get("citations") or [],
                                        note=args.get("reason_type", ""))
                state["violations"] = violations
                self._log(state, action, args, reason, "追问被拒：" + "；".join(violations))
                continue

            if action == "abstain":
                violations = self.guard.verify_abstain(args, state)
                if not violations:
                    self._log(state, action, args, reason, "缺口说明完整")
                    return self._finish(state, "insufficient_info",
                                        gaps=args.get("gaps") or [],
                                        note=f"已检索：{', '.join(state['queries'])}")
                state["violations"] = violations
                self._log(state, action, args, reason, "弃答被拒：" + "；".join(violations))
                continue

    # ---------- 内部 ----------
    def _build_evidence(self, state: dict) -> list[dict]:
        """把 hits / reads 摊平成模型可读的证据清单。不喂全文，避免上下文爆掉。"""
        items: list[dict] = []
        for doc_id, doc in state["reads"].items():
            for sec in doc["sections"]:
                items.append({
                    "doc_id": doc_id, "title": doc.get("title", ""),
                    "section_id": sec["section_id"], "heading": sec["heading"],
                    "version": doc.get("version", ""), "scope": doc.get("scope", ""),
                    "precedence": doc.get("precedence"), "status": doc.get("status", ""),
                    "text": sec["text"],
                })
        read_docs = set(state["reads"])
        for h in state["hits"]:
            if h["doc_id"] in read_docs:
                continue
            items.append({
                "doc_id": h["doc_id"], "title": h["title"], "section_id": h["section_id"],
                "heading": h["heading"], "version": h["version"], "scope": h["scope"],
                "precedence": h["precedence"], "status": h["status"], "text": h["snippet"],
            })
        return items[:8]

    def _init_state(self, question: str, history: list[dict]) -> dict:
        normalized = question
        if history:
            last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            # 指代消解：当前问题去掉停用词后实体词很少（如"我工作12年了"），
            # 才把上一轮问题拼进来；独立长问题不拼，避免污染查询。
            from ..retrieval.bm25 import STOPWORDS, tokenize
            content_terms = [t for t in tokenize(question) if t not in STOPWORDS]
            if last_user and len(content_terms) <= 2:
                normalized = f"{last_user}；{question}"
        scope = detect_scope(question) or detect_scope(normalized)
        sub_questions = [q.strip() for q in re.split(r"[，,；;]|以及", normalized) if q.strip()][:MAX_SUB_QUESTIONS]
        return {
            "question": question,
            "normalized": normalized,
            "history": history,
            "constraints": {"scope": scope},
            "sub_questions": sub_questions,
            "step": 0,
            "tool_calls": 0,
            "max_steps": MAX_STEPS,
            "max_tool_calls": MAX_TOOL_CALLS,
            "hits": [],
            "reads": {},
            "queries": [],
            "listed": False,
            "last_error": "",
            "last_observation": "",
            "seen": {},
            "repeats": 0,
            "dry": 0,
            "violations": [],
            "signals": {"conflicts": [], "versions": [], "scope": {"ambiguous": False, "values": []}},
        }

    def _exec(self, action: str, args: dict, state: dict) -> str:
        try:
            if action == "search":
                res = self.tools.search_docs(**args)
            elif action == "read":
                res = self.tools.read_doc(**args)
            else:
                res = self.tools.list_docs(**args)
        except ToolError as exc:
            state["last_error"] = f"{exc.code}: {exc.message}"
            state["last_observation"] = f"工具失败 {exc.code}：{exc.message}"
            return f"工具失败 {exc.code}（{exc.message}），建议：{exc.suggestion}"

        if action == "search":
            seen = {h["chunk_id"] for h in state["hits"]}
            for h in res["hits"]:
                if h["chunk_id"] not in seen:
                    state["hits"].append(h)
            state["queries"].append(args.get("query", ""))
            state["last_error"] = ""
            top = res["hits"][0]["chunk_id"] if res["hits"] else "-"
            digest = f"检索「{args.get('query','')}」命中 {res['total']} 条，最高 {top}"
            state["last_observation"] = digest
            state["dry"] = state.get("dry", 0) + 1 if res["total"] == 0 else 0
            return digest

        if action == "read":
            state["reads"][res["doc_id"]] = res
            state["last_error"] = ""
            secs = ",".join(s["section_id"] for s in res["sections"])
            digest = f"读取 {res['doc_id']} {res['version']}（{res['status']}），章节 {secs}"
            state["last_observation"] = digest
            state["dry"] = 0
            return digest

        state["listed"] = True
        digest = f"目录共 {res['total']} 篇：" + ",".join(d["doc_id"] for d in res["docs"][:8])
        state["last_observation"] = digest
        return digest

    def _log(self, state: dict, action: str, args: dict, reason: str, digest: str) -> None:
        self.events.append({
            "step": state["step"],
            "action": action,
            "args": args,
            "reason": reason,
            "observation_digest": digest,
            "signals": {
                "conflicts": [f"{c['slot']}:{'未解决' if c['unresolved'] else '可解决'}"
                              for c in state["signals"].get("conflicts", [])],
                "versions": [v["doc_id"] for v in state["signals"].get("versions", [])],
            },
            "ts": round(time.time(), 3),
        })

    def _finish(self, state: dict, status: str, answer: str = "", citations: list | None = None,
                gaps: list | None = None, note: str = "") -> Result:
        result = Result(question=state["question"], status=status, answer=answer,
                        citations=citations or [], gaps=gaps or [],
                        steps=state["step"], tool_calls=state["tool_calls"], note=note)
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("w", encoding="utf-8") as fh:
                for event in self.events:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
                fh.write(json.dumps({"step": state["step"], "action": "result",
                                     "status": status, "answer": answer[:200]},
                                    ensure_ascii=False) + "\n")
        return result


def answer_numbers(text: str) -> set[float]:
    return numbers(text)
