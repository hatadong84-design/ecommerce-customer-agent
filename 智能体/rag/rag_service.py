from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from 智能体.model.factory import chat_model
from 智能体.prompt_loader import load_rag_summarize_prompts
from 智能体.rag.vector_store import VectorStoreService

# 相关度门槛：检索分数低于该值视为「不相关」，不交给模型强行作答，
# 避免在知识库里根本没有资料时让模型凭常识编造答案。
RELEVANCE_THRESHOLD = 0.35

# 无可靠资料时的固定答复，明确说明查不到并给出后续处理方式
NO_KNOWLEDGE_ANSWER = (
    "知识库中没有找到与该问题相关的资料，暂时无法基于资料回答。"
    "你可以补充具体的型号或使用场景，或申请转人工客服，由人工协助处理。"
)

# 引用来源的标记：Web 层据此把来源随答案一起呈现给用户
RAG_SOURCE_PREFIX = "参考来源："


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_summarize_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def _source_of(self, doc) -> str:
        return doc.metadata.get("title") or doc.metadata.get("source") or "未知来源"

    def retriever_docs(self, query: str, category: str | None = None) -> list:
        """检索并按相关度门槛过滤；低相关的结果直接丢弃"""
        scored = self.vector_store.retrieve_with_scores(query, category=category)
        return [doc for doc, score in scored if score >= RELEVANCE_THRESHOLD]

    def rag_summarize(self, query: str, category: str = "") -> str:
        context_docs = self.retriever_docs(query, category or None)

        # 检索不到可靠资料时如实告知，并给出后续处理方式，不编造答案
        if not context_docs:
            return NO_KNOWLEDGE_ANSWER

        context = ""
        sources: list[str] = []
        for counter, doc in enumerate(context_docs, start=1):
            source = self._source_of(doc)
            context += f"【参考资料{counter}】：{doc.page_content}（来源：{source}）\n"
            if source not in sources:
                sources.append(source)

        answer = self.chain.invoke({"input": query, "context": context})
        # 回答末尾附上引用来源，让答案可追溯
        return f"{answer}\n\n{RAG_SOURCE_PREFIX}{'、'.join(sources)}"


if __name__ == "__main__":

    rag_summarize_service = RagSummarizeService()
    print(rag_summarize_service.rag_summarize("小型扫地机器人适合哪些场景"))
