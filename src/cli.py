"""命令行入口：建索引、问答、交互、重放。"""
from __future__ import annotations

import argparse
import json

from .agent import Agent
from .config import STATUS_ZH
from .index import build_index, load_index, save_index
from .llm import LLMClient, MockPolicy
from .retrieval import Retriever
from .tools import DocsTools


def build() -> None:
    index, vectors = build_index()
    save_index(index, vectors)
    print(f"索引已生成：{len(index['docs'])} 篇文档 / {len(index['chunks'])} 个 chunk，"
          f"向量后端={index['embedder']['kind']}:{index['embedder']['model']}")


def make_agent(fault: str | None = None, trace: str | None = None,
               mock: bool = False) -> Agent:
    index, vectors = load_index()
    tools = DocsTools(Retriever(index, vectors), fault_spec=fault)
    if mock:
        # 离线确定性策略：只验证链路，不证明真实模型决策质量
        return Agent(tools=tools, policy=MockPolicy(), trace_path=trace)
    client = LLMClient()
    if not client.available:
        raise SystemExit("未配置 LLM_API_KEY：请复制 .env.example 为 .env 填入密钥，"
                         "或加 --mock 用离线策略跑通链路")
    return Agent(tools=tools, policy=client, trace_path=trace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qa", description="企业知识问答 Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="重建索引")

    ask = sub.add_parser("ask", help="单轮提问")
    ask.add_argument("question")
    ask.add_argument("--trace", default=None)
    ask.add_argument("--fault", default=None)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--mock", action="store_true", help="离线确定性策略，无需模型密钥")

    chat = sub.add_parser("chat", help="多轮交互")
    chat.add_argument("--trace", default=None)
    chat.add_argument("--mock", action="store_true", help="离线确定性策略，无需模型密钥")

    replay = sub.add_parser("replay", help="重放一条 trace")
    replay.add_argument("path")

    args = parser.parse_args(argv)

    if args.cmd == "build":
        build()
        return 0

    if args.cmd == "replay":
        for line in open(args.path, encoding="utf-8"):
            event = json.loads(line)
            print(f"[{event['step']}] {event['action']:<7} {event.get('reason','')}")
            print(f"      {event.get('observation_digest','')}")
        return 0

    if args.cmd == "ask":
        agent = make_agent(args.fault, args.trace, mock=args.mock)
        result = agent.run(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"状态：{STATUS_ZH[result.status]}（{result.status}）")
            print(f"回答：{result.answer or '（无）'}")
            for c in result.citations:
                print(f"引用：{c['doc_id']} {c.get('section_id','')} 「{c['quote'][:40]}…」")
            if result.gaps:
                print("缺口：" + "；".join(result.gaps))
        return 0

    if args.cmd == "chat":
        agent = make_agent(None, args.trace, mock=args.mock)
        history: list[dict] = []
        print("输入问题，空行退出。")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q:
                break
            result = agent.run(q, history=history)
            print(f"[{STATUS_ZH[result.status]}] {result.answer or '（未给出答案）'}")
            history += [{"role": "user", "content": q},
                        {"role": "assistant", "content": result.answer or result.status}]
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


