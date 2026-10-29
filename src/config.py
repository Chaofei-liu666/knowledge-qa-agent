"""全局配置：路径、预算、模型接入参数。"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    """极简 .env 读取，避免为一个小项目引入 python-dotenv。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env(ROOT / ".env")

DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_PATH = DATA_DIR / "index.json"
VECTORS_PATH = DATA_DIR / "vectors.json"
SYNONYMS_PATH = DATA_DIR / "synonyms.json"
EXAMPLES_DIR = ROOT / "examples"

# 生效预算（题目要求：到上限必须返回明确的未完成状态）
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
MAX_SUB_QUESTIONS = int(os.getenv("MAX_SUB_QUESTIONS", "3"))

# 检索参数：两阶段 + RRF
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
# 第一阶段文档粗排：按相对阈值截断候选文档（不固定篇数）
# 保留 doc_score >= 最高分 * RATIO 的文档，至少 1 篇，最多 MAX 篇兜底
RETRIEVAL_DOC_RATIO = float(os.getenv("RETRIEVAL_DOC_RATIO", "0.30"))
RETRIEVAL_DOC_MAX = int(os.getenv("RETRIEVAL_DOC_MAX", "8"))
RETRIEVAL_RRF_K = int(os.getenv("RETRIEVAL_RRF_K", "60"))
# 相关性地板：BM25 无命中且余弦低于该值时，不把噪音当证据
RETRIEVAL_MIN_SEMANTIC_SCORE = float(os.getenv("RETRIEVAL_MIN_SEMANTIC_SCORE", "0.30"))

# 模型：OpenAI 兼容协议，默认指向 DeepSeek
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# 语义检索：有 embedding 接口就走接口，没有就用离线哈希向量兜底
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "256"))

# 故障注入：FAULT=search:1 表示第 1 次检索抛 index_unavailable
FAULT_SPEC = os.getenv("FAULT", "")

STATUSES = (
    "answered",           # 已回答
    "needs_clarification",  # 需澄清
    "insufficient_info",  # 资料不足
    "failed",             # 失败
    "incomplete",         # 未完成
)

STATUS_ZH = {
    "answered": "已回答",
    "needs_clarification": "需澄清",
    "insufficient_info": "资料不足",
    "failed": "失败",
    "incomplete": "未完成",
}

# 冲突检测用的槽位中文名（避免把英文键名暴露给用户）
SLOT_LABELS = {
    "reimbursement_deadline": "报销期限",
    "annual_leave_days": "年假天数",
    "password_rotation": "口令更换要求",
    "data_level": "数据分级",
}

# 适用范围关键词 → 规范值。语料范围变化时改这里即可，不必动 agent 代码。
SCOPE_KEYWORDS = {
    "境外": ["境外", "出国", "海外", "港澳"],
    "华东区": ["华东", "上海", "江苏", "浙江", "安徽"],
    "境内": ["境内", "国内"],
}

