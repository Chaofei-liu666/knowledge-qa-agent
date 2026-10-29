"""六种状态的端到端覆盖。"""
import pytest

from helpers import Scripted


def test_normal_answer(agent_factory):
    result = agent_factory().run("采购 30 万需要谁审批？")
    assert result.status == "answered"
    assert "分管副总" in result.answer
    assert result.citations[0]["doc_id"] == "PRC-001"


def test_conflict_asks_with_both_quotes(agent_factory):
    result = agent_factory().run("报销期限多久？")
    assert result.status == "needs_clarification"
    docs = {c["doc_id"] for c in result.citations}
    assert {"FIN-001", "FIN-004"} <= docs
    assert "30" in result.answer and "15" in result.answer


def test_precedence_resolves_conflict(agent_factory):
    result = agent_factory().run("华东区报销单提交时限是多久？")
    assert result.status == "answered"
    assert "30 个自然日" in result.answer
    assert {c["doc_id"] for c in result.citations} >= {"FIN-001", "FIN-002"}


def test_version_trap_uses_current_doc(agent_factory):
    result = agent_factory().run("密码多久换一次？")
    assert result.status == "answered"
    assert "MFA" in result.answer
    assert result.citations[0]["doc_id"] == "IT-001"
    assert any("已废止" in c["doc_id"] or c["doc_id"] == "IT-002" for c in result.citations)


def test_out_of_scope_reports_gap(agent_factory):
    result = agent_factory().run("期权行权价是多少？")
    assert result.status == "insufficient_info"
    assert result.gaps


def test_tool_failure_recovers(agent_factory):
    result = agent_factory(fault="search:1").run("采购 30 万需要谁审批？")
    assert result.status == "answered"


def test_budget_exhausted_is_incomplete(agent_factory, monkeypatch):
    monkeypatch.setattr("src.agent.MAX_STEPS", 2)
    policy = Scripted([{"action": "search", "args": {"query": "报销期限"}, "reason": "检索"}])
    result = agent_factory(policy=policy).run("报销期限多久？")
    assert result.status == "incomplete"
    assert "上限" in result.note


def test_abstain_needs_two_queries(agent_factory):
    """只搜一次就弃答会被拒绝。"""
    policy = Scripted([
        {"action": "search", "args": {"query": "期权"}, "reason": "检索"},
        {"action": "abstain", "args": {"gaps": ["没有"]}, "reason": "弃答"},
    ])
    result = agent_factory(policy=policy).run("期权行权价是多少？")
    assert result.status in {"incomplete", "insufficient_info"}
    assert result.status == "incomplete"
