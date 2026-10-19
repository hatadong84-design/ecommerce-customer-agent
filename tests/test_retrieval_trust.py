# -*- coding: utf-8 -*-
"""知识库检索可信度的回归测试：来源解析 + 低相关过滤 + 分类过滤。

不连真实向量库：用替身绕开 Chroma 与嵌入模型，专注验证门槛逻辑，
这样测试既快又不依赖网络，再去 eval/run_eval.py 做真实数据的评测。
"""
from langchain_core.documents import Document

from 智能体.agent.tools.agent_tools import RAG_SOURCE_PREFIX, extract_rag_sources
from 智能体.rag.rag_service import RELEVANCE_THRESHOLD, RagSummarizeService


# ---------------- 来源解析 ----------------
def test_extract_rag_sources_parses_titles():
    text = f"回答内容\n\n{RAG_SOURCE_PREFIX}扫地机械人、数据1"
    assert extract_rag_sources(text) == ["扫地机械人", "数据1"]


def test_extract_rag_sources_returns_empty_without_prefix():
    assert extract_rag_sources("普通回答") == []


def test_extract_rag_sources_handles_empty_input():
    assert extract_rag_sources("") == []
    assert extract_rag_sources(None) == []


# ---------------- 相关度门槛 ----------------
class _StubVectorStore:
    """向量库替身：记录调用参数并返回给定的 (文档, 分数)"""

    def __init__(self, scored):
        self.scored = scored
        self.calls = []

    def retrieve_with_scores(self, query, k=None, category=None):
        self.calls.append({"query": query, "k": k, "category": category})
        return self.scored


def _service(scored):
    # 用 __new__ 跳过 __init__，避免真的去连 Chroma 与嵌入模型
    service = RagSummarizeService.__new__(RagSummarizeService)
    service.vector_store = _StubVectorStore(scored)
    return service


def _doc(title):
    return Document(page_content="内容", metadata={"title": title})


def test_drops_documents_below_threshold():
    service = _service([(_doc("高相关"), 0.9), (_doc("低相关"), 0.1)])
    assert [d.metadata["title"] for d in service.retriever_docs("问题")] == ["高相关"]


def test_keeps_document_exactly_at_threshold():
    service = _service([(_doc("临界"), RELEVANCE_THRESHOLD)])
    assert len(service.retriever_docs("问题")) == 1


def test_returns_empty_when_all_below_threshold():
    """低相关时必须返回空，交给上层明确告知「没有相关资料」，而不是硬答"""
    service = _service([(_doc("无关"), 0.01)])
    assert service.retriever_docs("问题") == []


def test_category_filter_is_forwarded_to_vector_store():
    service = _service([(_doc("水杯资料"), 0.8)])
    service.retriever_docs("问题", category="水杯")
    assert service.vector_store.calls[0]["category"] == "水杯"
