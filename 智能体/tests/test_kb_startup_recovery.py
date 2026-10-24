from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from 智能体.rag import kb_service


class FakeLockConnection:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, statement, params):
        self.statements.append((statement, params))
        return self

    def fetchone(self):
        return None


def _patch_startup(monkeypatch, *, document_count, chunk_count, rebuild_error=None):
    docs = [{"id": 1, "index_status": kb_service.store.KB_INDEX_PENDING}]
    state = {"chunks": chunk_count, "rebuild_calls": 0}
    lock_connection = FakeLockConnection()

    monkeypatch.setattr(kb_service.store, "connect", lambda: lock_connection)
    monkeypatch.setattr(kb_service, "ensure_kb_seeded", lambda: None)
    monkeypatch.setattr(kb_service.store, "kb_document_total", lambda: document_count)
    monkeypatch.setattr(kb_service.store, "list_kb_documents", lambda: docs)

    def set_status(doc_id, status, error=None):
        row = next(doc for doc in docs if doc["id"] == doc_id)
        row["index_status"] = status
        row["index_error"] = error

    monkeypatch.setattr(kb_service.store, "set_document_index_status", set_status)

    class FakeVectorStore:
        def count_chunks(self):
            return state["chunks"]

    class FakeKnowledgeBaseService:
        def __init__(self):
            self.vector_store = FakeVectorStore()

        def rebuild_index(self):
            state["rebuild_calls"] += 1
            if rebuild_error:
                raise RuntimeError(rebuild_error)
            state["chunks"] = 4
            for row in docs:
                row["index_status"] = kb_service.store.KB_INDEX_DONE
            return len(docs)

    monkeypatch.setattr(kb_service, "KnowledgeBaseService", FakeKnowledgeBaseService)
    return state, docs, lock_connection


def test_startup_rebuilds_when_documents_exist_and_chroma_is_empty(monkeypatch):
    state, docs, lock = _patch_startup(
        monkeypatch, document_count=1, chunk_count=0
    )

    kb_service.ensure_kb_ready()

    assert state["rebuild_calls"] == 1
    assert state["chunks"] == 4
    assert docs[0]["index_status"] == kb_service.store.KB_INDEX_DONE
    assert "pg_advisory_lock" in lock.statements[0][0]
    assert "pg_advisory_unlock" in lock.statements[-1][0]


@pytest.mark.parametrize(
    ("document_count", "chunk_count"),
    [(0, 0), (1, 3)],
)
def test_startup_skips_rebuild_when_empty_kb_or_index_exists(
    monkeypatch, document_count, chunk_count
):
    state, _, _ = _patch_startup(
        monkeypatch, document_count=document_count, chunk_count=chunk_count
    )

    kb_service.ensure_kb_ready()

    assert state["rebuild_calls"] == 0


def test_rebuild_exception_marks_documents_failed_and_releases_lock(monkeypatch):
    _, docs, lock = _patch_startup(
        monkeypatch, document_count=1, chunk_count=0,
        rebuild_error="embedding service unavailable",
    )

    kb_service.ensure_kb_ready()

    assert docs[0]["index_status"] == kb_service.store.KB_INDEX_FAILED
    assert "RuntimeError" in docs[0]["index_error"]
    assert "pg_advisory_unlock" in lock.statements[-1][0]


def test_concurrent_startup_workers_only_rebuild_once(monkeypatch, store):
    state = {"chunks": 0, "rebuild_calls": 0}
    state_lock = threading.Lock()

    monkeypatch.setattr(kb_service, "ensure_kb_seeded", lambda: None)
    monkeypatch.setattr(kb_service.store, "kb_document_total", lambda: 1)
    monkeypatch.setattr(
        kb_service.store,
        "list_kb_documents",
        lambda: [{"id": 1, "index_status": kb_service.store.KB_INDEX_DONE}],
    )

    class FakeVectorStore:
        def count_chunks(self):
            with state_lock:
                return state["chunks"]

    class FakeKnowledgeBaseService:
        def __init__(self):
            self.vector_store = FakeVectorStore()

        def rebuild_index(self):
            with state_lock:
                state["rebuild_calls"] += 1
            time.sleep(0.1)
            with state_lock:
                state["chunks"] = 5
            return 1

    monkeypatch.setattr(kb_service, "KnowledgeBaseService", FakeKnowledgeBaseService)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(kb_service.ensure_kb_ready) for _ in range(2)]
        for future in futures:
            future.result(timeout=15)

    assert state["rebuild_calls"] == 1
    assert state["chunks"] == 5


def test_empty_vector_collection_with_documents_uses_unavailable_answer(monkeypatch):
    from 智能体.agent.answer_guard import apply_guard, prefetch_knowledge
    from 智能体.rag.rag_service import (
        KNOWLEDGE_UNAVAILABLE_ANSWER,
        RagSummarizeService,
    )

    class EmptyVectorStore:
        def retrieve_with_scores(self, query, category=None):
            return []

        def count_chunks(self):
            return 0

    monkeypatch.setattr(kb_service.store, "kb_document_total", lambda: 1)
    service = RagSummarizeService.__new__(RagSummarizeService)
    service.vector_store = EmptyVectorStore()

    knowledge = prefetch_knowledge(
        "扫地机器人噪音变大怎么办",
        categories=["扫地机器人"],
        rag=service,
    )

    assert knowledge["reason"] == "knowledge_service_unavailable"
    assert apply_guard("", knowledge) == KNOWLEDGE_UNAVAILABLE_ANSWER
