from src.retrieval import Retriever


def test_purchase_query_hits_prc(index):
    retriever = Retriever(*index)
    hits = retriever.search("采购 30 万需要谁审批？")
    assert hits and hits[0]["doc_id"] == "PRC-001"
    assert "5 万元" in hits[0]["snippet"]


def test_out_of_scope_returns_nothing(index):
    retriever = Retriever(*index)
    assert retriever.search("期权行权价是多少？") == []


def test_conflict_docs_both_retrieved(index):
    retriever = Retriever(*index)
    docs = {h["doc_id"] for h in retriever.search("报销期限多久？", top_k=8)}
    assert {"FIN-001", "FIN-004"} <= docs


def test_synonym_mapping(index):
    retriever = Retriever(*index)
    # 「多久内提交」应归一到「报销期限」，命中同一批文档
    hits = retriever.search("报销多久内提交？")
    assert {h["doc_id"] for h in hits} & {"FIN-001", "FIN-002", "FIN-004"}
