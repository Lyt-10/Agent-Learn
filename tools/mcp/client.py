"""MCP 客户端实现 - 使用 FastMCP"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

# 添加上级目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """MCP 客户端"""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.tools: list[dict] = []
        self._stack = AsyncExitStack()  # 托管上下文，让连接活到 close()

    async def connect_stdio(self, command: str, args: list[str] | None = None) -> None:
        """通过 stdio 连接到 MCP 服务器"""
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=None,
        )

        # 交给 stack 保管：不在函数结束时退出上下文，否则子进程会被立刻关掉
        read, write = await self._stack.enter_async_context(stdio_client(server_params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        self.session = session

        await session.initialize()

        # 获取可用工具列表
        tools_result = await session.list_tools()
        self.tools = [tool.model_dump() for tool in tools_result.tools]

    async def list_tools(self) -> list[dict]:
        """列出服务器上的所有工具"""
        if not self.session:
            raise RuntimeError("Not connected to server")
        tools_result = await self.session.list_tools()
        return [tool.model_dump() for tool in tools_result.tools]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """调用服务器上的工具"""
        if not self.session:
            raise RuntimeError("Not connected to server")
        result = await self.session.call_tool(name, arguments)
        return result

    async def close(self) -> None:
        """关闭连接"""
        await self._stack.aclose()  # 逆序退出：先关会话，再关子进程
        self.session = None


# 示例用法
async def main():
    client = MCPClient()

    # 连接到本地 MCP 服务器
    # 用绝对路径定位 server.py：从任何目录启动都能找到
    # 用 sys.executable 启动：保证服务器和客户端是同一个解释器（同一个 venv）
    server_script = Path(__file__).parent / "server.py"
    await client.connect_stdio(sys.executable, [str(server_script)])

    # 列出工具
    tools = await client.list_tools()
    print("Available tools:", [t["name"] for t in tools])

    # 调用工具
    result = await client.call_tool("add", {"a": 3, "b": 4})
    print("3 + 4 =", result)

    await client.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
