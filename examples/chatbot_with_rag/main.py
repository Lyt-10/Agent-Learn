"""Chatbot with RAG - 带知识库检索的对话机器人

使用方法:
    1. 在 .env 中配置 embedding 相关环境变量（EMBEDDING_MODEL_ID 等）
    2. 将 .txt / .md 文档放入 knowledge/ 目录
    3. 运行: python examples/chatbot_with_rag/main.py

启动时会自动索引 knowledge/ 目录下的所有文档，
对话中可询问文档相关内容，Agent 会自动调用 rag_search 工具检索。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm import call_llm
from core.node import Node, Flow, shared
from core.rag import RAGEngine
from tools import get_tools, ToolExecutor
from tools.builtins.rag import create_rag_tool

# ── 配置 ──
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

SYSTEM_PROMPT = """\
你是一个具有知识库检索能力的智能助手。
- 当用户询问关于已索引文档中的信息时，优先使用 rag_search 工具搜索知识库。
- 基于搜索结果回答用户问题，并引用来源文件路径。
- 如果知识库中没有相关信息，请如实告知用户，不要编造。
- 也可以使用其他工具（read, write, bash, search, grep, find, ls 等）辅助完成任务。
- 回答时尽量简洁、准确。
"""


# ── Node 定义 ──

class ChatNode(Node):
    """调用 LLM，打印回复，处理 tool_calls 路由。"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        messages = shared["messages"]
        tools = shared["tools"]

        assistant_message = call_llm(
            messages=messages,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        messages.append(assistant_message)

        content = assistant_message.get("content", "")
        if content:
            print(f"\n🤖 Assistant: {content}\n")

        if assistant_message.get("tool_calls"):
            return "tool_call", assistant_message

        return "done", assistant_message


class ToolCallNode(Node):
    """执行 LLM 返回的 tool_calls。"""

    def exec(self, payload: Any) -> Tuple[str, Any]:
        response = payload
        messages = shared["messages"]
        executor = shared["tool_executor"]

        tool_calls = executor.parse_tool_calls(response)
        results = executor.execute_all(tool_calls)

        for tc, result in zip(tool_calls, results):
            print(f"  🔧 {tc.name}({tc.arguments})")
            short = result.content[:150].replace("\n", " ")
            print(f"     → {short}...")
            messages.append(result.to_message())

        return "chat", None


# ── 启动逻辑 ──

def index_knowledge_base(engine: RAGEngine, knowledge_dir: Path) -> dict:
    """启动时索引知识库目录中的所有文档。"""
    if not knowledge_dir.exists():
        print(f"📁 知识库目录不存在: {knowledge_dir}")
        print("   创建目录并放入 .txt / .md 文件后重启即可。\n")
        return {}

    print(f"📂 索引知识库: {knowledge_dir}")
    result = engine.index_directory(str(knowledge_dir), glob_pattern="*.{txt,md}")

    total = sum(len(ids) for ids in result.values())
    print(f"   已索引 {len(result)} 个文件，共 {total} 个文档块")
    for file_path, doc_ids in result.items():
        print(f"   ├─ {file_path} ({len(doc_ids)} chunks)")

    existing = engine.count() - total
    if existing > 0:
        print(f"   └─ 已有 {existing} 个历史文档块（来自之前索引）")
    print()
    return result


def run_chat() -> None:
    """初始化并运行 RAG 对话循环。"""
    print("=" * 60)
    print("🤖 Chatbot with RAG — 带知识库检索的对话机器人")
    print("=" * 60)

    # ── 初始化 RAG 引擎 ──
    print("\n🔧 初始化知识库引擎...")
    engine = RAGEngine(collection_name="tutorial_kb")
    existing_count = engine.count()
    if existing_count > 0:
        print(f"   已加载历史知识库: {existing_count} 个文档块")

    # ── 索引新文档 ──
    index_knowledge_base(engine, KNOWLEDGE_DIR)

    # ── 装配工具 ──
    rag_tool = create_rag_tool(engine)
    executor = ToolExecutor()
    executor.tools.append(rag_tool)
    executor.tool_map[rag_tool.name] = rag_tool

    shared.clear()
    shared["messages"] = []
    shared["tools"] = [t.to_llm_format() for t in get_tools()]
    shared["tools"].append(rag_tool.to_llm_format())
    shared["tool_executor"] = executor

    # ── 构建 Node 图 ──
    chat = ChatNode()
    tool_call = ToolCallNode()
    chat - "tool_call" >> tool_call
    tool_call - "chat" >> chat

    print("可用工具: read, write, edit, bash, grep, find, ls, search, rag_search")
    print("输入 'quit' 或 'exit' 退出\n")

    # ── 对话循环 ──
    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        if not user_input:
            continue

        shared["messages"].append({"role": "user", "content": user_input})
        Flow(chat).run(None)


def main() -> None:
    """入口：检查环境变量后启动对话。"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  请先设置 OPENAI_API_KEY（在 .env 文件中）")
        return
    if not os.environ.get("OPENAI_BASE_URL"):
        print("⚠️  请先设置 OPENAI_BASE_URL（在 .env 文件中）")
        return

    run_chat()


if __name__ == "__main__":
    main()
