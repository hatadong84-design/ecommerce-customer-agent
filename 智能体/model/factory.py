import os
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from 智能体.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            temperature=0,
            timeout=60,
            max_retries=2,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            # SiliconFlow 只接受纯文本输入和 float 向量：
            # 默认的 token 数组输入与 base64 编码会返回 400（code 20015）
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
        )


chat_model = ChatModelFactory().generator()
embeddings_model = EmbeddingsFactory().generator()
