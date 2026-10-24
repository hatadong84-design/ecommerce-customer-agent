from pathlib import Path

from 智能体.rag.vector_store import VectorStoreService

QUESTIONS = [
    "扫地机器人和扫拖一体机有什么区别？",  # 基础认知
    "激光导航和视觉导航哪个好？",          # 选购指南
    "扫地机器人无法回充怎么办？",          # 故障排查
    "滚刷被头发缠住了怎么清理？",          # 保养维护
    "一套耗材大概用多久？",                # 耗材成本
]

if __name__ == "__main__":
    vs = VectorStoreService()

    total = vs.count_chunks()
    print(f"向量库中当前共有 {total} 个文档片段")
    if total == 0:
        print("库是空的：请先运行 python -m 智能体.rag.vector_store 完成加载")
        raise SystemExit(1)

    retriever = vs.get_retriever()
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 40}\n【问题{i}】{q}")
        docs = retriever.invoke(q)
        if not docs:
            print("  没有检索到任何片段")
            continue
        for j, d in enumerate(docs, 1):
            snippet = " ".join(d.page_content.split())[:60]
            source = Path(d.metadata.get("source", "未知")).name
            print(f"  片段{j}（来源: {source}）: {snippet}...")
