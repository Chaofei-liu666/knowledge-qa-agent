"""引用闸门与冲突闸门：几种必须拒绝的错误答案。"""
from helpers import Scripted


def _search(query):
    return {"action": "search", "args": {"query": query}, "reason": "检索"}


def test_fabricated_quote_rejected(agent_factory):
    bad = {"action": "answer", "args": {
        "answer": "报销期限是 15 个自然日。",
        "citations": [{"doc_id": "FIN-001", "section_id": "5.2",
                       "quote": "报销期限是 15 个自然日。"}]}, "reason": "答"}
    policy = Scripted([
        _search("报销期限多久"),
        {"action": "read", "args": {"doc_id": "FIN-001"}, "reason": "读原文"},
        bad, bad,
    ])
    result = agent_factory(policy=policy).run("报销期限多久？")
    assert result.status == "incomplete"
    assert any("原文逐字" in g for g in result.gaps)


def test_superseded_only_citation_rejected(agent_factory):
    bad = {"action": "answer", "args": {
        "answer": "口令每 90 天必须更换一次。",
        "citations": [{"doc_id": "IT-002", "section_id": "4.1",
                       "quote": "员工口令每 90 天必须更换一次，口令长度不得少于 8 位，且三个月内不得重复使用历史口令。"}]},
        "reason": "答"}
    policy = Scripted([_search("密码多久换一次"), bad, bad])
    result = agent_factory(policy=policy).run("密码多久换一次？")
    assert result.status == "incomplete"
    assert any("已废止" in g for g in result.gaps)


def test_unresolved_conflict_single_answer_rejected(agent_factory):
    bad = {"action": "answer", "args": {
        "answer": "请在 15 个工作日内提交。",
        "citations": [{"doc_id": "FIN-004", "section_id": "Q7",
                       "quote": "答：请在 15 个工作日内提交，以免影响当月结算。"}]},
        "reason": "答"}
    policy = Scripted([_search("报销期限多久"), bad, bad])
    result = agent_factory(policy=policy).run("报销期限多久？")
    assert result.status == "incomplete"
    assert any("冲突未解释" in g for g in result.gaps)


def test_answer_beyond_quote_rejected(agent_factory):
    bad = {"action": "answer", "args": {
        "answer": "单笔 5 万元以上由分管副总审批，报销比例 80%。",
        "citations": [{"doc_id": "PRC-001", "section_id": "2.1",
                       "quote": "单笔采购金额 5 万元以下的，由部门负责人审批；5 万元（含）至 50 万元的，由分管副总审批；50 万元（含）以上的，由总经理审批。"}]},
        "reason": "答"}
    policy = Scripted([_search("采购审批"), bad, bad])
    result = agent_factory(policy=policy).run("采购 30 万需要谁审批？")
    assert result.status == "incomplete"
    assert any("超出引用范围" in g for g in result.gaps)
