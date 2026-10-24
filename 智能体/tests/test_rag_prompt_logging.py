import logging

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from 智能体.rag import rag_service


class FakeVectorStore:
    def __init__(self, document):
        self.document = document

    def count_chunks(self):
        return 1

    def retrieve_with_scores(self, query, category=None):
        return [(self.document, 0.95)]


def test_rag_chain_passes_prompt_to_model_without_writing_it_to_logs(capsys, caplog):
    query = "敏感用户问题_不应进入日志"
    source_text = "敏感知识库正文_不应进入日志"
    seen_prompts = []
    service = rag_service.RagSummarizeService.__new__(rag_service.RagSummarizeService)
    service.vector_store = FakeVectorStore(Document(
        page_content=source_text,
        metadata={"title": "测试来源"},
    ))
    service.prompt_template = PromptTemplate.from_template(
        "问题：{input}\n参考资料：{context}"
    )
    service.model = RunnableLambda(
        lambda prompt: (
            seen_prompts.append(prompt.to_string()),
            AIMessage(content="依据资料生成的回答"),
        )[1]
    )
    service.chain = service._init_chain()

    with caplog.at_level(logging.DEBUG):
        answer = service.rag_summarize(query)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert query not in caplog.text
    assert source_text not in caplog.text
    assert query in seen_prompts[0]
    assert source_text in seen_prompts[0]
    assert "依据资料生成的回答" in answer
    assert "参考来源：测试来源" in answer
    assert not hasattr(rag_service, "print_prompt")
