# 企业知识问答 Agent

企业制度文档问答 CLI。输入 markdown 制度文件，输出带原文引用和版本段落定位的答案；
多文件数值冲突时列出双方证据请求确认；检索不到返回资料缺口。

## 快速开始

```bash
pip install -r requirements.txt
python -m src.build_index
python -m src.cli ask "采购 30 万需要谁审批？" --mock    # 零配置离线跑通
# 接真实模型：cp .env.example .env 填入 LLM_API_KEY 后去掉 --mock
```

## 怎么看这个项目

ReAct 循环每一步的动作、参数、决策依据、工具返回摘要都落在 `examples/*/trace.jsonl`，
`python -m src.cli replay <trace.jsonl>` 可逐步重放。六条用例覆盖：

| 用例 | 场景 | trace |
|---|---|---|
| case01 | 明确问题直接作答 | `examples/case01_normal/trace.jsonl` |
| case02 | 两份文件数值打架 → 追问用户 | `examples/case02_conflict_ask/trace.jsonl` |
| case03 | 有优先级的冲突 → 按优先级作答 | `examples/case03_conflict_resolved/` |
| case04 | 超纲 → 报缺口 | `examples/case04_insufficient/` |
| case05 | 检索故障 → 模型改调 list_docs → 重试后作答 | `examples/case05_tool_failure/trace.jsonl` |
| case06 | 多轮追问补全省略词 | `examples/case06_multiturn/` |

`--mock` 是离线确定性策略，只验证链路，不证明真实模型决策质量；
真实模型（deepseek-chat）跑的记录见 `examples/`，`input.json` 里 `provider: llm`。

## 命令

| 命令 | 说明 |
|---|---|
| `python -m src.cli ask "<问题>"` | 单轮问答（需模型密钥） |
| `python -m src.cli ask "<问题>" --mock` | 离线确定性策略，无需密钥 |
| `python -m src.cli chat` | 多轮交互（可加 `--mock`） |
| `python -m src.cli ask "<问题>" --json` | 结构化输出 |
| `python -m src.cli ask "<问题>" --fault "search:1"` | 注入第 1 次检索故障 |
| `python -m src.cli replay <trace.jsonl>` | 重放运行记录 |
| `pytest -q` | 23 条测试，其中 6 条是拒绝路径 |

## 结构

```
src/agent/      ReAct 循环、状态机、四道 answer 校验、trace 落盘
src/retrieval/  两阶段召回：文档摘要粗排（BM25+向量 RRF，阈值截断）→ chunk 精排
src/tools/      search_docs / read_doc / list_docs，统一 ToolError
src/llm/        OpenAI 兼容客户端、系统提示词、MockPolicy
data/docs/      12 篇合成制度（1 篇废止、3 篇报销期限数值冲突）
examples/       6 组输入、实际输出与逐步 trace
docs/           需求 / 方案 / 评测 / AI 使用 / 复盘
```

## 文档

- [需求与验收标准](docs/01-requirements.md)
- [调用流程、动态决策、技术取舍](docs/02-solution.md)
- [测试结果、运行记录、失败与限制](docs/03-evaluation.md)
- [AI 辅助内容与人工修改](docs/04-ai-usage.md)
- [最难问题、学习收获、下一步](docs/05-retrospective.md)

