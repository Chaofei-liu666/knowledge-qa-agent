"""重新生成 examples/ 下的运行记录：python scripts/run_examples.py

每条用例写 input.json / output.json / trace.jsonl，可直接复核决策链。
需要先配置 .env 中的模型密钥。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import Agent  # noqa: E402
from src.index import load_index  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.tools import DocsTools  # noqa: E402

PROVIDER = "llm"
POLICY = LLMClient()
if not POLICY.available:
    raise SystemExit("未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填入密钥")

CASES = [
    ("case01_normal", "采购 30 万需要谁审批？", None),
    ("case02_conflict_ask", "报销期限多久？", None),
    ("case03_conflict_resolved", "华东区报销单提交时限是多久？", None),
    ("case04_insufficient", "期权行权价是多少？", None),
    ("case05_tool_failure", "采购 30 万需要谁审批？", "search:1"),
]


def _agent(fault: str | None, trace: Path) -> Agent:
    index, vectors = load_index()
    return Agent(tools=DocsTools(Retriever(index, vectors), fault_spec=fault),
                 policy=POLICY, trace_path=trace)


def run_single(name: str, question: str, fault: str | None) -> None:
    out = ROOT / "examples" / name
    out.mkdir(parents=True, exist_ok=True)
    result = _agent(fault, out / "trace.jsonl").run(question)
    (out / "input.json").write_text(
        json.dumps({"question": question, "history": [], "fault": fault, "provider": PROVIDER},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "output.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"{name}: {result.status}")


def run_multiturn() -> None:
    out = ROOT / "examples" / "case06_multiturn"
    out.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    results = []
    for i, question in enumerate(["年假有几天？", "我工作 12 年了"], start=1):
        result = _agent(None, out / f"trace_turn{i}.jsonl").run(question, history=history)
        results.append(result.to_dict())
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": result.answer or result.status}]
        print(f"case06 turn{i}: {result.status}")
    (out / "input.json").write_text(json.dumps(
        {"turns": ["年假有几天？", "我工作 12 年了"], "note": "第二轮省略了「年假」"}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "output.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                     encoding="utf-8")


if __name__ == "__main__":
    for name, question, fault in CASES:
        run_single(name, question, fault)
    run_multiturn()
