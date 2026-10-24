import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from 智能体.config_handler import chroma_conf
from 智能体.model.factory import embeddings_model
from 智能体.path_toll import get_abs_path
from 智能体.file_hander import (
    listdir_with_allowed_type,
    get_file_md5_hex,
    pdf_loader,
    txt_loader,
)
from 智能体.logger_hander import logger

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embeddings_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def count_chunks(self) -> int:
        """返回当前集合中的向量块数，封装 Chroma 集合的内部访问。"""
        return self.vector_store._collection.count()

    def retrieve_with_scores(self, query: str, k: int | None = None,
                             category: str | None = None) -> list[tuple[Document, float]]:
        """带相关度分数的检索（分数越高越相关）。
        category 非空时只在对应商品分类内检索，避免不同商品的资料混在一起。"""
        kwargs: dict = {"k": k or chroma_conf["k"]}
        if category:
            kwargs["filter"] = {"category": category}
        return self.vector_store.similarity_search_with_relevance_scores(query, **kwargs)

    # ---------- 以下是知识库管理用到的按文档增删改索引 ----------
    def index_document(self, doc_id: int, category: str, title: str, text: str) -> int:
        """切分一篇知识文档并写入向量库；先清旧分片再写入，所以更新即重索引。
        分片带 doc_id 元数据，这样按文档删除/更新时能精确定位到属于它的分片。"""
        self.delete_document(doc_id)
        chunks = self.spliter.create_documents(
            [text],
            metadatas=[{"doc_id": str(doc_id), "category": category, "title": title}],
        )
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)

    def delete_document(self, doc_id: int) -> None:
        """按 doc_id 删除该文档在向量库里的全部分片"""
        try:
            self.vector_store.delete(where={"doc_id": str(doc_id)})
        except Exception as e:
            logger.error(f"[知识库]删除文档 {doc_id} 的向量分片失败: {str(e)}")

    def reset_index(self) -> None:
        """清空整个向量集合，供 kb_service 按 kb_document 全量重建"""
        self.vector_store.reset_collection()

    def check_md5_hex(self, md5_for_check: str):
        if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
            # 创建文件
            open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
            return False  # md5 没处理过

        with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line == md5_for_check:
                    return True  # md5 处理过

        return False  # md5 没处理过

    def save_md5_hex(self, md5_for_check: str):
        with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
            f.write(md5_for_check + "\n")

    def get_file_documents(self, read_path: str):
        if read_path.endswith("txt"):
            return txt_loader(read_path)

        if read_path.endswith("pdf"):
            return pdf_loader(read_path)

        return []

    def load_document(self):
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)
            if self.check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = self.get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                self.save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)

if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("扫地机器人怎么保养")   # 测试查询建议改成和知识库相关的内容
    for r in res:
        print(r.page_content)
        print("-"*20)



