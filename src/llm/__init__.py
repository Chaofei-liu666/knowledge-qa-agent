"""模型接入。

两种实现共用同一个决策协议（JSON 动作），所以 mock 能直接跑测试与演示；
但 mock 只证明链路能走通，不证明真实模型的决策质量，README 与 03 里都标了这条边界。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from ..config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_MODEL, LLM_TIMEOUT, SLOT_LABELS
from ..tools import ToolError  # noqa: F401  (供调用方捕获)


class LLMError(Exception):
    pass


SYSTEM_PROMPT = """你是企业知识问答 Agent。事实只能来自工具返回的原文，不得凭记忆作答。

每一步只输出一个 JSON 决策：
{"action": "search|read|answer|ask|abstain", "args": {...}, "reason": "不超过30字"}

动作与参数：
- search:  {"query": "...", "top_k": 5, "scope": 可选}
- read:    {"doc_id": "FIN-001", "section_id": 可选}
- list:    {}  列目录，确认语料范围（判断"没有这类资料"时必须先调用）
- answer:  {"answer": "...", "citations": [{"doc_id": "...", "section_id": "...", "quote": "原文逐字"}]}
- ask:     {"question": "...", "citations": [...], "reason_type": "conflict|scope|missing_entity"}
- abstain: {"gaps": ["..."], "reason": "..."}

规则：
1. 每条引用必须逐字来自本会话工具返回过的原文，不得改写、拼接或引用未返回的内容。
2. 两份文件对同一件事给出不同数值，且都没有适用范围或优先级时，必须 ask：列出双方原句与文档 ID，请用户确认适用制度，不得自行选择。
3. 存在优先级、版本替代或适用范围时，给出确定结论，并引用支撑该结论的原文。
4. 已废止（superseded）文档不得作为唯一依据。
5. 检索返回 0 条时，不要反复换同义词：先 list_docs 确认语料范围；确认没有后 abstain，
   并在 gaps 里列出已检索的关键词。至少两次不同检索 + 一次 list_docs 才允许弃答。
6. 已有证据时不要重复同样的检索；先读原文或直接作答，重复调用不会带来新信息。
7. reason 只写决策依据，不写推理过程。
8. 只输出 JSON，不要代码块，不要多余文字。"""


class LLMClient:
    """OpenAI 兼容协议。没有 key 时 available=False。"""

    name = "llm"

    def __init__(self) -> None:
        self.model = LLM_MODEL
        self.available = bool(LLM_API_KEY)

    def decide(self, state: dict) -> dict:
        if not self.available:
            raise LLMError("未配置 LLM_API_KEY")
        user = render_state(state)
        content = self._chat(SYSTEM_PROMPT, user)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 一次修复：保留系统提示词（动作枚举/规则不能丢），只在用户消息末尾追加纠错
            content = self._chat(
                SYSTEM_PROMPT,
                user + "\n\n上一条输出不是合法 JSON，请严格按动作协议重新给出一个 JSON，不要代码块。")
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMError(f"模型输出不是合法 JSON：{content[:120]}") from exc

    def _chat(self, system: str, user: str, attempt: int = 0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_API_KEY}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            if attempt < LLM_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 4))
                return self._chat(system, user, attempt + 1)
            raise LLMError(f"模型调用失败（重试 {LLM_MAX_RETRIES} 次后）：{exc}") from exc
        return data["choices"][0]["message"]["content"]


def render_state(state: dict) -> str:
    lines = [f"用户问题：{state['normalized']}"]
    if state.get("history"):
        lines.append("对话历史：" + " | ".join(f"{h['role']}:{h['content'][:60]}" for h in state["history"][-4:]))
    if state.get("constraints"):
        lines.append(f"已抽取的约束：{state['constraints']}")
    if state.get("sub_questions"):
        lines.append("子问题：" + "；".join(state["sub_questions"]))
    lines.append(f"预算：已用 {state['step']}/{state['max_steps']} 步，工具调用 {state['tool_calls']}/{state['max_tool_calls']}")

    if state.get("evidence"):
        lines.append("已获取的证据：")
        for e in state["evidence"][:8]:
            lines.append(f"- {e['doc_id']} {e['section_id']} {e['heading']}（{e['version']}，{e['scope']}，"
                         f"优先级 {e['precedence']}，状态 {e['status']}）：{e['text'][:120]}")
    else:
        lines.append("已获取的证据：无")

    sig = state.get("signals") or {}
    parts = []
    for c in sig.get("conflicts", []):
        vals = "；".join(f"{v['doc_id']}={v['value']}" for v in c["values"])
        parts.append(f"冲突[{c['slot']}]{'未解决' if c['unresolved'] else '可解决'}：{vals}")
    for v in sig.get("versions", []):
        parts.append(f"版本[{v['doc_id']}]已被 {v['superseded_by']} 替代")
    if sig.get("scope", {}).get("ambiguous"):
        parts.append("范围不明确：" + "、".join(sig["scope"]["values"]))
    lines.append("检测到的信号：" + ("；".join(parts) if parts else "无"))
    if state.get("last_observation"):
        lines.append(f"上一次工具结果：{state['last_observation']}")
    if state.get("dry", 0) >= 2:
        lines.append("警告：检索连续 0 条，语料里很可能没有这类资料。不要再检索、也不要再列目录："
                     "现在直接 abstain，gaps 写清检索过的关键词。")
    if state.get("dry", 0) >= 3:
        lines.append("强制要求：下一步只能是 abstain（或 ask，如果你判断缺的是条件而不是资料）。")
    if state.get("last_error"):
        lines.append(f"上一次工具失败：{state['last_error']}")
    if state.get("repeats"):
        lines.append(f"注意：同样的动作与参数已重复 {state['repeats']} 次，结果不会变。"
                     f"请立刻改用 answer / ask / abstain，不要再调用工具。")
    if state.get("violations"):
        lines.append(f"上一次回答被校验拒绝：{state['violations']}")
    return "\n".join(lines)


def _alt_query(text: str) -> str:
    """换一种说法重搜：去掉疑问词与标点，尽量保留实体词。"""
    q = re.sub(r"[？?。，,！!；;：:]+", "", text)
    q = re.sub(r"(是多少|多少|怎么样|怎么|如何|请问|一下)", "", q)
    return q or f"{text} 制度"


class MockPolicy:
    """离线确定性策略：按 state 的观察结果决定动作。

    它不是"正确模型"的替身，只是把链路跑通：search → read → answer/ask/abstain。
    """

    name = "mock"
    available = True

    def decide(self, state: dict) -> dict:
        sig = state.get("signals") or {}
        hits = state.get("hits") or []
        reads = state.get("reads") or {}
        scope = (state.get("constraints") or {}).get("scope")

        if not state["queries"]:
            return {"action": "search",
                    "args": {"query": state["normalized"], "top_k": 5, "scope": scope},
                    "reason": "先做一次召回"}

        if state.get("last_error") and len(state["queries"]) < 2:
            return {"action": "search",
                    "args": {"query": state["normalized"], "top_k": 5},
                    "reason": "检索失败，重试并放宽范围"}

        if not hits and not state.get("listed"):
            return {"action": "list", "args": {}, "reason": "无命中，先确认语料范围"}

        if not hits:
            if len(set(state["queries"])) < 2:
                return {"action": "search",
                        "args": {"query": _alt_query(state["normalized"]), "top_k": 5, "scope": scope},
                        "reason": "换一组说法再确认一次"}
            return {"action": "abstain",
                    "args": {"gaps": ["语料中未收录该问题相关的制度或 FAQ"],
                             "searched": state["queries"]},
                    "reason": "两次检索加目录确认仍无资料"}

        conflict = next((c for c in sig.get("conflicts", []) if c["unresolved"]), None)
        if conflict:
            for value in conflict["values"]:
                doc_id = value["doc_id"]
                if doc_id not in reads:
                    return {"action": "read", "args": {"doc_id": doc_id},
                            "reason": "两份数值冲突，读原文查适用范围与优先级"}
            if not state.get("asked_conflict"):
                return self._ask_conflict(conflict)

        if sig.get("scope", {}).get("ambiguous") and not state.get("asked_scope"):
            return {"action": "ask",
                    "args": {"question": f"请问是{'还是'.join(sig['scope']['values'])}的情况？适用范围不同，结论也不同。",
                             "citations": self._scope_citations(state),
                             "reason_type": "scope"},
                    "reason": "适用范围不明确，先澄清"}

        resolved = next((c for c in sig.get("conflicts", [])
                         if not c["unresolved"] and c.get("winner")), None)
        if resolved:
            winner = resolved["winner"]
            loser_value = next(v for v in resolved["values"] if v["doc_id"] != winner)
            if winner not in reads:
                return {"action": "read", "args": {"doc_id": winner},
                        "reason": "两份数值冲突，按优先级读优先文件"}
            if loser_value["doc_id"] not in reads:
                return {"action": "read", "args": {"doc_id": loser_value["doc_id"]},
                        "reason": "读对照文件，确认冲突原句"}
            winner_value = next(v for v in resolved["values"] if v["doc_id"] == winner)
            return self._answer_from(state, winner, extra=loser_value,
                                     section_id=winner_value["section_id"])

        # 版本替代：命中的是废止文档时，必须换到现行版本再答
        top = hits[0]
        if top["status"] == "superseded" and top.get("superseded_by"):
            replacement = top["superseded_by"]
            if replacement not in reads:
                return {"action": "read", "args": {"doc_id": replacement},
                        "reason": "命中的是废止版本，改读现行版本"}
            return self._answer_from(state, replacement, note_old=top["doc_id"])

        if top["doc_id"] not in reads:
            return {"action": "read", "args": {"doc_id": top["doc_id"], "section_id": top["section_id"]},
                    "reason": "命中片段不足以作答，读原文确认"}

        return self._answer_from(state, top["doc_id"])

    # ---- 组装动作 ----
    def _answer_from(self, state: dict, doc_id: str, note_old: str | None = None,
                     extra: dict | None = None, section_id: str | None = None) -> dict:
        doc = state["reads"][doc_id]
        if not section_id:
            cands = [h for h in state["hits"] if h["doc_id"] == doc_id]
            hit = max(cands, key=lambda h: h["bm25"]) if cands else None
            section_id = hit["section_id"] if hit else doc["sections"][0]["section_id"]
        section = next((s for s in doc["sections"] if s["section_id"] == section_id), doc["sections"][0])
        body = section["text"].split("。")
        quote = "。".join(body[:2]).strip()
        if not quote.endswith("。"):
            quote += "。"

        answer = f"根据《{doc['title']}》（{doc['version']}）：{quote}"
        if note_old:
            answer += f" 另：{note_old} 已被替代，不再适用。"
        citations = [{"doc_id": doc_id, "section_id": section["section_id"], "quote": quote}]
        if note_old:
            old_section, old_quote = self._old_ref(state, note_old)
            citations.append({"doc_id": note_old, "section_id": old_section, "quote": old_quote})
        if extra:
            answer += f" 另：{extra['doc_id']} 的规定与此不一致，按优先级以本文件为准。"
            citations.append({"doc_id": extra["doc_id"], "section_id": extra["section_id"],
                              "quote": extra["quote"]})
        return {"action": "answer", "args": {"answer": answer, "citations": citations},
                "reason": "证据足够，给出结论并附引用"}

    @staticmethod
    def _old_ref(state: dict, doc_id: str) -> tuple[str, str]:
        doc = state["reads"].get(doc_id)
        if doc and doc["sections"]:
            return doc["sections"][0]["section_id"], doc["sections"][0]["text"][:80]
        for h in state["hits"]:
            if h["doc_id"] == doc_id:
                return h["section_id"], h["snippet"][:80]
        return "", ""

    def _ask_conflict(self, conflict: dict) -> dict:
        values = conflict["values"]
        label = SLOT_LABELS.get(conflict["slot"], conflict["slot"])
        desc = "；".join(f"{v['doc_id']} 写的是 {v['value']}" for v in values)
        return {"action": "ask",
                "args": {"question": f"关于{label}，材料里的说法不一致：{desc}。"
                                     f"它们没有共同的适用范围或优先级，请确认按哪份执行？",
                         "citations": [{"doc_id": v["doc_id"], "section_id": v["section_id"],
                                        "quote": v["quote"]} for v in values],
                         "reason_type": "conflict"},
                "reason": "冲突且无优先级，先确认适用制度"}

    def _scope_citations(self, state: dict) -> list[dict]:
        out = []
        for h in state["hits"][:3]:
            if h["status"] == "active":
                out.append({"doc_id": h["doc_id"], "section_id": h["section_id"], "quote": h["snippet"][:80]})
        return out
