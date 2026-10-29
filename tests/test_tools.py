"""工具契约与错误语义。"""
import pytest

from src.retrieval import Retriever
from src.tools import DocsTools, ToolError


@pytest.fixture()
def tools(index):
    return DocsTools(Retriever(*index))


def test_unknown_doc_raises_not_found(tools):
    with pytest.raises(ToolError) as exc:
        tools.read_doc("FIN-999")
    assert exc.value.code == "not_found"
    assert "list_docs" in exc.value.suggestion


def test_empty_query_raises_invalid_args(tools):
    with pytest.raises(ToolError) as exc:
        tools.search_docs("")
    assert exc.value.code == "invalid_args"


def test_injected_fault(index):
    faulty = DocsTools(Retriever(*index), fault_spec="search:1")
    with pytest.raises(ToolError) as exc:
        faulty.search_docs("报销期限")
    assert exc.value.code == "index_unavailable"
    # 第二次应恢复正常
    assert faulty.search_docs("报销期限")["total"] >= 1


def test_list_docs_counts(tools):
    data = tools.list_docs()
    assert data["total"] >= 10
    assert {"FIN-001", "HR-001", "IT-001"} <= {d["doc_id"] for d in data["docs"]}


def test_read_section_filters(tools):
    data = tools.read_doc("FIN-001", section_id="5.2")
    assert [s["section_id"] for s in data["sections"]] == ["5.2"]
    assert "30 个自然日" in data["sections"][0]["text"]
