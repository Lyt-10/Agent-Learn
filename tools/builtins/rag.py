# tools/builtins/rag.py
"""RAG 检索工具：在本地向量库中查找与问题相关的文档块。"""

from __future__ import annotations

from functools import lru_cache

MAX_HITS_IN_OUTPUT = 5       # 一次最多回几块
MAX_CHARS_PER_HIT = 700      # 每块最多回多少字，防止上下文被检索结果冲爆


@lru_cache(maxsize=1)
def _get_store():
    """懒加载单例：只有真的调用 rag_search 时才连库、才校验 API key。"""
    from core.rag.store import RagStore

    return RagStore()


def rag_search(query: str, top_k: int = 5, source: str = "") -> str:
    """在本地知识库（向量库）中检索与 query 最相关的文档片段。

    Args:
        query: 自然语言问题或关键词。
        top_k: 返回条数，建议 3~5。
        source: 可选，只在这个文件路径（metadata.source）里检索。
    """
    try:
        store = _get_store()
    except Exception as exc:                     # noqa: BLE001 —— 错误也是信息，交给模型判断
        return f"Error: 向量库不可用（{exc}）。请先执行：uv run python -m core.rag.index <路径>"

    if store.collection.count() == 0:
        return "知识库为空。请先运行索引命令，或改用 grep/find 在本仓库中查找。"

    hits = store.search(
        query=query,
        top_k=min(max(int(top_k), 1), MAX_HITS_IN_OUTPUT),
        source=source or None,
    )
    if not hits:
        return f"没有检索到与「{query}」相关的内容。"

    blocks = []
    for hit in hits:
        text = hit.text if len(hit.text) <= MAX_CHARS_PER_HIT else hit.text[:MAX_CHARS_PER_HIT] + "…"
        blocks.append(
            f"【片段 {len(blocks) + 1}】来源: {hit.source} (chunk {hit.chunk_index}, "
            f"相似度 {hit.score:.3f})\n{text}"
        )
    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(rag_search("上下文太长怎么办"))