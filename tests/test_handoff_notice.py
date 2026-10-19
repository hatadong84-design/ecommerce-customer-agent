# -*- coding: utf-8 -*-
"""转人工判定与「工单已提交」告知文案的回归测试。

这两段逻辑直接决定用户能不能看到「工单已提交」，
以及退款类诉求会不会被漏掉不去建单（业务要求：退款一律转人工）。
"""
from 智能体.agent.tools.agent_tools import RAG_SOURCE_PREFIX
from 智能体.api.app import handoff_notice, needs_ticket, with_handoff_notice, with_rag_sources


# ---------------- 是否建单 ----------------
def test_needs_ticket_when_sentiment_flags_human():
    assert needs_ticket({"need_human": True, "category": "售后"}) is True


def test_needs_ticket_for_refund_category_even_if_flag_false():
    """退款类诉求是硬规则：即使情绪分析没标 need_human，也必须转人工"""
    assert needs_ticket({"need_human": False, "category": "退款"}) is True


def test_needs_ticket_false_for_normal_sales_question():
    assert needs_ticket({"need_human": False, "category": "售前"}) is False
    assert needs_ticket({}) is False


# ---------------- 告知文案 ----------------
def test_notice_says_submitted_for_new_ticket():
    text = handoff_notice({"id": 12, "status": "待接入", "created": True})
    assert "已为您提交" in text
    assert "#12" in text
    assert "待接入" in text


def test_notice_says_updated_for_reused_ticket():
    text = handoff_notice({"id": 9, "status": "处理中", "created": False})
    assert "已为您更新" in text
    assert "#9" in text


def test_notice_is_empty_without_ticket():
    assert handoff_notice(None) == ""


def test_with_handoff_notice_appends_to_answer():
    out = with_handoff_notice("好的，已记录。", {"id": 3, "status": "待接入", "created": True})
    assert out.startswith("好的，已记录。")
    assert "【人工工单】" in out


def test_with_handoff_notice_is_idempotent():
    ticket = {"id": 3, "status": "待接入", "created": True}
    once = with_handoff_notice("好的", ticket)
    assert with_handoff_notice(once, ticket) == once, "重复追加会生成两条工单通知"


def test_with_handoff_notice_leaves_answer_when_no_ticket():
    assert with_handoff_notice("好的", None) == "好的"


# ---------------- 引用来源 ----------------
def test_with_rag_sources_appends_prefix():
    out = with_rag_sources("这是回答", ["扫地机械人"])
    assert RAG_SOURCE_PREFIX in out
    assert "扫地机械人" in out


def test_with_rag_sources_skips_duplicate():
    text = f"这是回答\n\n{RAG_SOURCE_PREFIX}扫地机械人"
    assert with_rag_sources(text, ["其它资料"]) == text


def test_with_rag_sources_without_sources():
    assert with_rag_sources("这是回答", []) == "这是回答"
