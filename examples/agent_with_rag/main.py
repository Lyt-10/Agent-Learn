"""Agent with RAG - 带本地知识库检索的工具 Agent

与 chatbot_with_memory 的差别只有三处：SYSTEM_PROMPT、启动提示、知识库健康检查。
Node / Flow 的图完全一致 —— 因为 RAG 只是「多了一个工具」，不是一套新框架。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm import call_llm
from core.node import Node, Flow, shared
from core.memory import Memory
from tools import get_tools, ToolExecutor

SYSTEM_PROMPT = (
    "你是一个会调用工具的助手。"
    "回答关于本项目文档、代码、设计的问题时，优先调用 rag_search 检索本地知识库，"
    "再基于检索到的片段回答，并在回答里标注来源文件（例如 core/memory.py）。"
    "检索结果不足以回答时，再用 grep/read 直接查看文件。"
    "当问题涉及最新信息、模型版本、产品发布时间或事实核验时，改用 search 联网搜索。"
    "如果一轮回复中既需要向用户展示文字又需要继续调用工具，可以同时返回 content 和 tool_calls。"
)


class ChatNode(Node):
    """发送消息给 LLM，获取响应（可能包含 tool_calls）"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        memory = shared["memory"]
        tools = shared["tools"]

        messages = memory.build_context(system_prompt=SYSTEM_PROMPT)
        assistant_message = call_llm(messages=messages, tools=tools)
        memory.add_message(assistant_message)

        if assistant_message.get("tool_calls"):
            return "tool_call", assistant_message

        return "output", assistant_message


class ToolCallNode(Node):
    """执行 LLM 返回的 tool_calls"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        response = payload
        memory = shared["memory"]
        executor = shared["tool_executor"]

        tool_calls = executor.parse_tool_calls(response)
        results = executor.execute_all(tool_calls)

        for tool_call, result in zip(tool_calls, results):
            print(f"  [Tool] 执行: {tool_call.name}({tool_call.arguments})")
            print(f"  [Tool] 结果: {result.content[:100]}...")
            memory.add_message(result.to_message())

        return "chat", None


class OutputNode(Node):
    """输出助手回复"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        response = payload
        content = response.get("content", "")
        print(f"\n🤖 Assistant: {content}\n")
        return "default", None


def describe_knowledge_base() -> str:
    """启动时探一眼知识库，让人一眼分清「没建库」和「检索没命中」。

    注意：这里必须 try 住。没配 EMBEDDING_* 或 chromadb 不可用时，
    不能因为探库失败就让人连对话都进不去 —— 与 rag_search 里
    「错误是信息，不是崩溃」的处理保持一致。
    """
    try:
        from core.rag.store import RagStore

        stats = RagStore().stats()
    except Exception as exc:
        return f"知识库不可用（{exc}）\n   → 请检查 .env 里的 EMBEDDING_API_KEY / EMBEDDING_BASE_URL"

    if not stats["chunks"]:
        return (
            "知识库为空\n"
            "   → 请先建库：python -m core.rag.index ./README.md ./learn.md ./core ./tools"
        )

    return f"知识库：{stats['files']} 个文件 / {stats['chunks']} 块 @ {stats['persist_dir']}"


def run_chat() -> None:
    """运行对话循环"""
    # 工具列表从注册表里读，别写死 —— 以后再加工具这里不会过期
    tool_names = ", ".join(t.name for t in get_tools())

    print("=" * 60)
    print("🤖 Agent with RAG")
    print("=" * 60)
    print(f"可用工具: {tool_names}")
    print(f"📦 {describe_knowledge_base()}")
    print("记忆管理: 短期上下文 + 长期记忆 (自动压缩)")
    print("输入 'quit' 或 'exit' 退出\n")

    shared.clear()

    shared["memory"] = Memory()
    shared["tools"] = [t.to_llm_format() for t in get_tools()]
    shared["tool_executor"] = ToolExecutor()

    chat = ChatNode()
    tool_call = ToolCallNode()
    output = OutputNode()

    chat - "tool_call" >> tool_call
    tool_call - "chat" >> chat
    chat - "output" >> output

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！")
            break

        if not user_input:
            continue

        shared["memory"].add_message({"role": "user", "content": user_input})
        flow = Flow(chat)
        flow.run(None)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        print("⚠️  提示：请先设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL")
        return

    run_chat()


if __name__ == "__main__":
     main()
