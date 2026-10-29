# 02 方案

## 模块划分

| 模块 | 文件 | 职责 | 为什么这么分 |
|---|---|---|---|
| 语料解析 | `src/corpus.py` | 解析 front matter，按 `##` 标题切节 | 切块粒度对齐条款号，引用才能定位到 5.2 |
| 索引构建 | `src/index.py` | 生成 chunk、文档摘要、BM25 统计、向量文件 | 摘要单独嵌入，供第一阶段文档粗排 |
| 检索 | `src/retrieval/` | 分词、BM25、embedding、两阶段召回与 RRF | 纯检索，不碰对话状态，可单测 |
| 工具 | `src/tools/__init__.py` | 检索 / 读原文 / 列目录，统一 ToolError | 工具是模型能做的事，错误必须转成 observation 而不是抛穿 |
| 模型接入 | `src/llm/__init__.py` | OpenAI 兼容客户端、系统提示词、状态渲染 | 真实模型与 MockPolicy 共用同一 decide 协议 |
| 决策循环 | `src/agent/__init__.py` | 状态机、Guard 校验、trace 落盘 | 程序只执行和校验，不替模型选动作 |
| 入口 | `src/cli.py` | build / ask / chat / replay | 所有命令走同一套构建与循环，--mock 和 --fault 只改 LLM 与工具层，业务逻辑不动 |

## 数据模型

### 文档 front matter

```yaml
id: FIN-001
title: 差旅与费用报销管理办法
version: v3.2
effective_date: 2025-03-01
scope: 集团
precedence: 1            # 数字越小优先级越高；空表示未标注
status: active           # active | superseded
superseded_by:           # 被替代时指向新文档
topics: 差旅,报销
slots: reimbursement_deadline=30 自然日
```

### chunk

```python
{
  "chunk_id": "FIN-001#5.2",     # doc_id#section_id
  "doc_id": "FIN-001", "title": "...", "version": "v3.2",
  "effective_date": "2025-03-01", "scope": "集团",
  "precedence": 1, "status": "active", "superseded_by": None,
  "topics": ["差旅", "报销"], "slots": {"reimbursement_deadline": "30 自然日"},
  "section_id": "5.2", "heading": "5.2 报销时限", "text": "员工应在出差结束后……"
}
```

### 索引文件

`data/index.json`：`built_at`、`embedder{kind, model}`、`docs[]`、`chunks[]`、
`bm25{k1, b, n, avgdl, df, tf, doclen}`。
`data/vectors.json`：`{chunk_id: [float]}`，另含 `__doc__:<doc_id>` 文档摘要向量，供第一阶段粗排。

## 工具接口

```python
search_docs(query, top_k=5, scope=None, include_superseded=True) -> dict
read_doc(doc_id, section_id=None) -> dict
list_docs(topic=None) -> dict
```

`search_docs` 返回命中列表，每条含 doc_id、版本、章节、摘要、`score`/`bm25`/`cosine`。
`read_doc` 返回文档元信息与指定章节（缺省返回全文各节）。
`list_docs` 返回文档清单，用于确认语料范围。

异常统一为 `ToolError(code, message, suggestion)`：

| code | 触发条件 |
|---|---|
| `invalid_args` | query 为空、top_k 越界、doc_id 缺失 |
| `not_found` | 文档或章节不存在 |
| `index_unavailable` | 故障注入或检索后端不可用 |

## 检索

两阶段。离线入库时为每篇文档生成摘要（标题 + 各节标题与首句）并单独嵌入，
存入 `vectors["__doc__:<doc_id>"]`。

第一阶段文档粗排：对每篇可见文档，BM25 分取其 chunk 最大值，余弦取 query 与
文档摘要向量的相似度；两路各自排名后按 RRF（k=60）融合成文档分，再按相对阈值
截断——保留文档分不低于最高分 30% 的文档，至少 1 篇、最多 8 篇兜底，**不固定篇数**。
这样纯语义相关、关键词零命中的文档也能进候选。

第二阶段 chunk 精排：只在候选文档内，对 chunk 用 BM25 与余弦各排一次序，按 RRF 合并。
过滤条件为 `scope ∈ {约束值, 集团, 全集团}`，`include_superseded` 控制是否排除废止文档。

相关性下限按向量后端取值：哈希 0.30、HTTP embedding 0.45、本地 bge 0.55。
这三个值是在 12 篇语料上跑若干典型问题后，按得分分布人工选定的，
未做系统调参；语料扩大后需要重标。低于下限且 BM25 为 0 的片段不进入候选。

术语表 `data/synonyms.json` 把同义说法映射到标准词，索引与查询两侧都执行映射。

## 决策循环

状态字段：`question`、`normalized`、`constraints`、`sub_questions`、`hits`、`reads`、
`evidence`、`signals`、`queries`、`seen`、`step`、`tool_calls`。

```python
while step < MAX_STEPS and tool_calls < MAX_TOOL_CALLS:
    state["evidence"] = build_evidence(state)
    decision = policy.decide(state)          # {"action", "args", "reason"}
    action = decision["action"]

    if action in {search, read, list}:
        tool_calls += 1
        observation = execute(action, args)   # ToolError 也转成 observation
        state["signals"] = guard.scan(state)
    elif action == answer:
        violations = guard.verify_answer(args, state)
        if not violations:
            return answered
        state["violations"] = violations          # 不通过则回灌原因
    elif action == ask:
        return needs_clarification
    elif action == abstain:
        return insufficient_info

return incomplete   # 步数或工具调用到顶
```

同一动作与参数重复超过两次时跳过执行，将该次调用标记为"重复调用"，并写入上下文。

### 状态与信号

程序侧维护一份运行时状态，每步渲染给模型看：

- `hits` / `reads`：本轮检索到的片段和已读全文；`evidence` 是两者摊平后的前 8 条，
  不喂全文是为了控上下文长度。
- `signals`：Guard 从已收集证据里扫出的结构化信号——
  `conflicts`（同槽位多文档取值不一致，标 unresolved 或带 winner）、
  `versions`（命中的是废止文档）、`scope`（适用范围是否歧义）。
- `queries`：已检索过的查询串，用于弃答门槛（至少两次不同检索）。
- `dry`：连续 0 命中次数，≥2 时提示模型别再搜、直接 abstain。
- `violations`：上一次 answer/ask/abstain 被拒的原因，回灌给模型让它改。

信号只影响模型看到的上下文，不直接改动作——这是"规则不越界"的关键：
模型看到"冲突未解决"信号后，自己决定是 ask 还是按优先级 answer。

### 终止条件

- 模型选 `answer` 且四道 answer 校验全过（引用、废止、冲突、数量）→ 已回答；
- 模型选 `ask` 且追问校验过 → 需澄清（等用户补充）；
- 模型选 `abstain` 且弃答门槛过 → 资料不足；
- 连续两次 answer 被闸门拒 → 未完成（把拒因写进 gaps）；
- 步数 ≥10 或工具调用 ≥8 → 未完成（附带已收集证据，不返回空结果）；
- 模型调用连续失败 → 失败。

## 提示词

为什么用 JSON 动作协议而不是原生 function calling：换模型（DeepSeek / Qwen / GPT）
不用改代码，测试里的 Scripted 策略和真实模型共用同一个 `decide(state) -> dict` 接口。
代价是模型偶发输出非法 JSON，所以解析失败时保留原系统提示词、在用户消息末尾追加纠错
重试一次，再失败才报 LLMError。

系统提示词给出动作枚举、每个动作的参数结构、八条规则（引用逐字、冲突须问、废止文档
不可单独引用、两次检索后才可弃答、重复调用无意义、只输出 JSON）。
每步渲染状态：问题、约束、子问题、预算、证据清单、信号、上次工具结果、重复与校验提示。

## 校验

answer 前跑四道校验，ask 与 abstain 各有一道门槛；失败把原因写回上下文让模型重写，连续两次不过才终止：

| 校验 | 判据 | 拦的是什么错误 |
|---|---|---|
| 引用逐字 | `quote` 归一化后是本轮证据原文的子串；`doc_id` 在本轮 hits/reads 里 | 模型凭记忆编引用、引用上一轮对话的旧文档 |
| 废止 | 引用不能全部来自 `superseded` 文档 | 模型答了已废止版本的规定 |
| 冲突 | 无优先级冲突时，双方 doc_id 必须同时出现在 citations 里 | 模型看到两份文件打架却只答一方 |
| 数量 | answer 里带单位的数字必须出现在引用或原问题里 | 模型编一个引用里没有的数字（如"报销比例 80%"） |
| 追问 | 冲突追问须引用双方；其他追问须给 `reason_type` | 模型无理由地反问用户 |
| 弃答 | 至少两次不同检索、一次 list_docs、`gaps` 非空 | 模型搜一次没命中就甩锅"没有资料" |

## 配置

| 变量 | 默认 | 作用 |
|---|---|---|
| `MAX_STEPS` | 10 | 决策步数上限 |
| `MAX_TOOL_CALLS` | 8 | 工具调用上限 |
| `RETRIEVAL_TOP_K` | 5 | 返回片段数 |
| `RETRIEVAL_DOC_RATIO` | 0.30 | 第一阶段候选文档保留比例（相对最高分） |
| `RETRIEVAL_DOC_MAX` | 8 | 第一阶段候选文档上限（兜底） |
| `RETRIEVAL_RRF_K` | 60 | RRF 平滑常数 |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | — | 模型接入 |
| `EMBED_PROVIDER` | 空（哈希） | 选择向量后端：`api` 走 HTTP 接口，`fastembed` 用本地 bge |
