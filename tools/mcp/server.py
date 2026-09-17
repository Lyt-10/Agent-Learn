"""MCP 服务器实现 - 使用 FastMCP"""

from __future__ import annotations

import sys
from pathlib import Path

# 被客户端当子进程 spawn 时，sys.path[0] 是「本文件所在目录」(tools/mcp)，
# 里面没有 tools 包。把项目根目录加进来，工具函数里的 `from tools...import`
# 才能成功。
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastmcp import FastMCP


# 创建 FastMCP 实例
mcp = FastMCP("agent-tools")


@mcp.tool()
def search(query: str, max_results: int = 5) -> list[dict]:
    """使用 DuckDuckGo 搜索网页"""
    from tools.builtins.search import search as search_impl
    return search_impl(query, max_results)


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


# 示例用法
if __name__ == "__main__":
    # 直接运行服务器 (stdio 传输)
    mcp.run(transport="stdio")
