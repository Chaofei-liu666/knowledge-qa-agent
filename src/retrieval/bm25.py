"""分词与 BM25。

中文用 jieba 精确模式 + 领域词表；域外词（同义改写）靠 synonyms.json 在查询侧归一到同一组词。
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import jieba  # noqa: E402

jieba.setLogLevel(logging.ERROR)

from ..config import SYNONYMS_PATH  # noqa: E402

STOPWORDS = set(
    "的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "这 那 它 他 她 与 或 及 等 中 对 为 从 被 把 而 但 并 很 更 最 可以 需要 请问 多少".split()
)

_jieba_ready = False


def _ensure_dict() -> None:
    global _jieba_ready
    if _jieba_ready:
        return
    for w in ["报销时限", "报销期限", "年休假", "试用期", "多因素认证", "数据分级",
              "医疗期", "供应商准入", "调休", "门禁卡", "餐卡", "汇率", "审批权限"]:
        jieba.add_word(w)
    _jieba_ready = True


def load_synonyms(path: Path | None = None) -> dict[str, str]:
    """variant -> canonical，例如「多久内提交」->「报销期限」。"""
    path = path or SYNONYMS_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for canonical, variants in data.items():
        for v in variants:
            mapping[v] = canonical
    return mapping


def normalize_query(query: str, synonyms: dict[str, str]) -> str:
    """术语归一到查询侧：把同义说法替换成检索词表中的标准词。"""
    out = query
    for variant in sorted(synonyms, key=len, reverse=True):
        if variant and variant in out:
            out = out.replace(variant, synonyms[variant])
    return out


def tokenize(text: str) -> list[str]:
    _ensure_dict()
    tokens = []
    for tok in jieba.lcut_for_search(text.lower()):
        tok = tok.strip()
        if not tok or tok in STOPWORDS:
            continue
        if all(not ch.isalnum() for ch in tok):
            continue
        tokens.append(tok)
    return tokens


class BM25:
    """Okapi BM25。k1=1.5, b=0.75，参数固定便于复现。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.doclen: dict[str, int] = {}
        self.df: Counter = Counter()
        self.n = 0
        self.avgdl = 0.0

    def fit(self, tokenized: dict[str, list[str]]) -> "BM25":
        self.tf = {cid: Counter(toks) for cid, toks in tokenized.items()}
        self.doclen = {cid: len(toks) for cid, toks in tokenized.items()}
        self.n = len(self.tf)
        self.avgdl = (sum(self.doclen.values()) / self.n) if self.n else 0.0
        self.df = Counter()
        for toks in tokenized.values():
            for term in set(toks):
                self.df[term] += 1
        return self

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, cid: str, query_tokens: list[str]) -> float:
        tf = self.tf.get(cid, Counter())
        dl = self.doclen.get(cid, 0) or 1
        total = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total += self.idf(term) * f * (self.k1 + 1) / denom
        return total

    def to_dict(self) -> dict:
        return {
            "k1": self.k1, "b": self.b, "n": self.n, "avgdl": self.avgdl,
            "df": dict(self.df),
            "tf": {cid: dict(counter) for cid, counter in self.tf.items()},
            "doclen": self.doclen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25":
        obj = cls(data.get("k1", 1.5), data.get("b", 0.75))
        obj.n = data["n"]
        obj.avgdl = data["avgdl"]
        obj.df = Counter(data["df"])
        obj.doclen = data["doclen"]
        obj.tf = {cid: Counter(counts) for cid, counts in data["tf"].items()}
        return obj
