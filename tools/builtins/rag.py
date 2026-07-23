"""RAG 搜索工具 — 将 RAGEngine 包装为 Agent 可调用的 Tool。

使用示例:
    from core.rag import RAGEngine
    from tools.builtins.rag import create_rag_tool

    engine = RAGEngine(collection_name="my_docs")
    rag_tool = create_rag_tool(engine)
"""

from __future__ import annotations

from typing import Any

from tools.builtins.tool_def import Tool


def create_rag_tool(engine: Any) -> Tool:
    """创建一个绑定到指定 RAGEngine 实例的 rag_search 工具。

    遵循 agent_with_goal 中 make_goal_complete_tool 的闭包模式。

    Args:
        engine: RAGEngine 实例。

    Returns:
        配置好的 Tool 对象。
    """

    def rag_search(query: str, top_k: int = 5) -> str:
        """搜索本地知识库，返回相关文档片段。"""
        return engine.query(query, top_k=top_k)

    return Tool(
        name="rag_search",
        description=(
            "搜索本地知识库（RAG），查找与查询相关的文档。"
            "当用户询问已索引文档中的信息时使用此工具。"
            "返回按相关度排序的结果，包含来源文件路径和内容摘要。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索知识库的查询语句",
                },
                "top_k": {
                    "type": "integer",
                    "description": "最多返回的结果数（默认 5）",
                },
            },
            "required": ["query"],
        },
        fn=rag_search,
    )
