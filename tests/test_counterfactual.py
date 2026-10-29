"""反事实测试：改文档内容，答案必须跟着变；删掉冲突文档，必须退回单来源作答。"""
import shutil
from pathlib import Path

from src.agent import Agent
from src.config import DOCS_DIR
from src.corpus import load_docs
from src.index import build_index
from src.llm import MockPolicy
from src.retrieval import Retriever
from src.tools import DocsTools


def _agent(docs_dir: Path) -> Agent:
    index, vectors = build_index(load_docs(docs_dir))
    return Agent(tools=DocsTools(Retriever(index, vectors)), policy=MockPolicy())


def _copy_corpus(tmp_path: Path) -> Path:
    dst = tmp_path / "docs"
    shutil.copytree(DOCS_DIR, dst)
    return dst


def test_answer_follows_document_change(tmp_path):
    docs_dir = _copy_corpus(tmp_path)
    path = docs_dir / "FIN-001.md"
    # 同步改正文与 front matter 的 slots（冲突检测读 slots），避免误改其它数字
    text = path.read_text(encoding="utf-8")
    text = text.replace("出差结束后 30 个自然日内", "出差结束后 45 个自然日内")
    text = text.replace("reimbursement_deadline=30 自然日", "reimbursement_deadline=45 自然日")
    path.write_text(text, encoding="utf-8")

    result = _agent(docs_dir).run("报销期限多久？")
    assert "45 自然日" in result.answer
    assert "30 个自然日" not in result.answer


def test_remove_conflicting_docs_answers_directly(tmp_path):
    """参考实例最后一句：只剩 D01、无冲突时，可依据 D01 回答。"""
    docs_dir = _copy_corpus(tmp_path)
    (docs_dir / "FIN-004.md").unlink()
    (docs_dir / "FIN-002.md").unlink()

    result = _agent(docs_dir).run("报销期限多久？")
    assert result.status == "answered"
    assert "30 个自然日" in result.answer
    assert result.citations[0]["doc_id"] == "FIN-001"
