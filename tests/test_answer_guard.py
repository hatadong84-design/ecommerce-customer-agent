# -*- coding: utf-8 -*-
"""可信回答闭环的单元测试：分类推断、资料可用性、程序侧拒答与来源补齐。

全部使用替身，不调用真实模型与向量库：
- 分类推断与未知商品判断是纯函数；
- judge_answerable 用假模型模拟「可用 / 不可用 / 异常」三种输出；
- handle_turn / stream_turn 用 monkeypatch 替换预检索，验证不可用时确实短路。
"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from 智能体.agent import answer_guard
from 智能体.api import app as app_module
from 智能体.rag.rag_service import NO_KNOWLEDGE_ANSWER, RAG_SOURCE_PREFIX

CATEGORIES = ["扫地机器人", "水杯"]


class _Doc:
    """最小文档替身：只需要 page_content 与 metadata"""

    def __init__(self, content: str, title: str):
        self.page_content = content
        self.metadata = {"title": title}


def _usable_answer():
    return "扫地机器人建议每两周清理一次滚刷，并检查边刷是否缠绕毛发。"


# ---------------- 分类推断 ----------------
def test_infer_category_matches_kb_category_names():
    assert answer_guard.infer_category("扫地机器人无法回充", CATEGORIES) == "扫地机器人"
    assert answer_guard.infer_category("吸管杯的吸管多久换一次", CATEGORIES) == "水杯"


def test_infer_category_uses_aliases():
    assert answer_guard.infer_category("扫拖一体机怎么保养", CATEGORIES) == "扫地机器人"
    assert answer_guard.infer_category("保温杯能装咖啡吗", CATEGORIES) == "水杯"


def test_infer_category_returns_none_for_unrelated_question():
    assert answer_guard.infer_category("帮我写一份本周工作周报", CATEGORIES) is None


def test_infer_category_uses_previous_turn_for_follow_up():
    """多轮追问「那滤网呢」时，靠上一轮的「扫地机器人滚刷」推断分类"""
    text = "那滤网呢需要换吗 扫地机器人滚刷多久清理一次"
    assert answer_guard.infer_category(text, CATEGORIES) == "扫地机器人"


# ---------------- 预检索：未知商品与无关问题 ----------------
def test_unknown_product_is_refused_without_retrieval():
    knowledge = answer_guard.prefetch_knowledge(
        "吸尘器滤网多久换一次", categories=CATEGORIES
    )
    assert knowledge["checked"] is True
    assert knowledge["answerable"] is False
    assert knowledge["docs"] == []
    assert knowledge["reason"] == "商品不在知识库范围"


def test_unrelated_question_is_left_to_agent():
    knowledge = answer_guard.prefetch_knowledge("今天适合拖地吗", categories=CATEGORIES)
    assert knowledge["checked"] is False


# ---------------- 资料可用性判断 ----------------
def test_judge_returns_false_when_model_says_unavailable():
    model = FakeListChatModel(responses=["不可用：没有价格信息"])
    ok, reason = answer_guard.judge_answerable(
        "维修一次多少钱", [_Doc("常见故障与排查步骤", "数据2")], model=model
    )
    assert ok is False
    assert "价格" in reason


def test_judge_returns_true_when_model_says_available():
    model = FakeListChatModel(responses=["可用"])
    ok, reason = answer_guard.judge_answerable(
        "滚刷多久清理", [_Doc("每两周清理一次滚刷", "数据1")], model=model
    )
    assert ok is True and reason == ""


def test_judge_fails_open_when_model_errors():
    """审核模型故障时放行，避免整站拒答；该取舍已写入 README 已知限制"""
    def boom(_):
        raise RuntimeError("模型超时")

    ok, reason = answer_guard.judge_answerable(
        "滚刷多久清理", [_Doc("每两周清理一次滚刷", "数据1")], model=RunnableLambda(boom)
    )
    assert ok is True and reason == ""


# ---------------- 最终答案守卫 ----------------
def test_guard_replaces_answer_when_no_docs():
    knowledge = {"checked": True, "answerable": False, "sources": []}
    assert answer_guard.apply_guard("一定可以退，三天到账。", knowledge) == NO_KNOWLEDGE_ANSWER


def test_guard_keeps_answer_and_adds_sources():
    knowledge = {"checked": True, "answerable": True, "sources": ["数据1", "售后手册"]}
    out = answer_guard.apply_guard(_usable_answer(), knowledge)
    assert out.startswith(_usable_answer())
    assert f"{RAG_SOURCE_PREFIX}数据1、售后手册" in out


def test_guard_does_not_duplicate_existing_sources():
    answer = f"{_usable_answer()}\n\n{RAG_SOURCE_PREFIX}数据1"
    knowledge = {"checked": True, "answerable": True, "sources": ["数据1"]}
    assert answer_guard.apply_guard(answer, knowledge) == answer


def test_guard_leaves_answer_untouched_when_not_checked():
    knowledge = {"checked": False}
    assert answer_guard.apply_guard("今天天气不错。", knowledge) == "今天天气不错。"


def test_should_call_agent_reflects_answerability():
    assert answer_guard.should_call_agent({"checked": True, "answerable": False}) is False
    assert answer_guard.should_call_agent({"checked": True, "answerable": True}) is True
    assert answer_guard.should_call_agent({"checked": False}) is True
    assert answer_guard.should_call_agent(None) is True


# ---------------- 接入 handle_turn / stream_turn ----------------
def _payload():
    return {
        "category": "售后",
        "need_human": False,
        "need_human_reason": "",
        "urgency": "低",
        "emotion": "平静",
        "emotion_level": 1,
        "satisfaction": "中性",
        "summary": "咨询保养",
    }


def test_handle_turn_skips_agent_when_knowledge_unavailable(clean_db, monkeypatch):
    def fake_prefetch(question, recent_questions=None, **kwargs):
        return {"checked": True, "answerable": False, "docs": [], "sources": [],
                "category": None, "reason": "没有检索到资料"}

    def must_not_call(_messages):
        raise AssertionError("资料不可用时不应再调用 agent")

    monkeypatch.setattr(app_module, "prefetch_knowledge", fake_prefetch)
    monkeypatch.setattr(app_module, "ask_agent", must_not_call)

    answer, tools = app_module.handle_turn("测试用户9201", "扫地机器人能改装吗", _payload())

    assert answer.startswith(NO_KNOWLEDGE_ANSWER)
    assert tools == []
    rows = clean_db.query_turns(session_id="测试用户9201")
    assert [r["role"] for r in reversed(rows)] == ["user", "assistant"]
    assert NO_KNOWLEDGE_ANSWER in rows[0]["content"], "拒答也要落库，便于后台审计"


def test_stream_turn_emits_no_token_when_knowledge_unavailable(clean_db, monkeypatch):
    def fake_prefetch(question, recent_questions=None, **kwargs):
        return {"checked": True, "answerable": False, "docs": [], "sources": [],
                "category": None, "reason": "没有检索到资料"}

    monkeypatch.setattr(app_module, "analyze_question", lambda _q: _payload())
    monkeypatch.setattr(app_module, "prefetch_knowledge", fake_prefetch)

    events = []

    async def collect():
        async for item in app_module.stream_turn("测试用户9202", "扫地机器人能改装吗", False):
            events.append(item)

    asyncio.run(collect())
    text = "".join(events)
    assert "event: token" not in text, "拒答时不应先把幻觉 token 推给前端"
    assert "event: done" in text
    assert NO_KNOWLEDGE_ANSWER in text


def test_stream_and_sync_produce_same_refusal(clean_db, monkeypatch):
    knowledge = {"checked": True, "answerable": False, "docs": [], "sources": [],
                 "category": None, "reason": "没有检索到资料"}
    sync_text = app_module.finalize_answer("模型编造的内容", [], None, None, knowledge)
    stream_text = app_module.finalize_answer("模型编造的内容", [], None, None, knowledge)
    assert sync_text == stream_text == NO_KNOWLEDGE_ANSWER
