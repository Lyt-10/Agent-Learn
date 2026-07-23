# Agent K 项目简介

## 什么是 Agent K？

Agent K 是一个从零开始学习 Agent 开发的教程项目。核心理念是：**Agent 其实非常简单！**

## 核心概念

### Node（节点）
Node 是 Agent 的最小执行单元。每个 Node 接收输入，执行逻辑，输出结果和下一步的路由。

```
输入 → Node.exec() → (action, 输出)
```

### Flow（流程）
Flow 把多个 Node 串联起来，根据每个 Node 返回的 action 决定下一个要执行的 Node。

### Workflow = Node + Node
把两个 Node 串起来就是一个 Workflow。例如：`接收输入 → 上网搜索 → 大模型生成总结`

### Chatbot = Workflow + Loop
在 Workflow 外面包一层循环，就是聊天机器人。

### Agent = Chatbot + Tools
给 Chatbot 加上工具调用能力，就是 Agent。工具让 Agent 可以读文件、写文件、搜索网页、执行命令等。

## 技术栈

- **Python 3.13+**
- **OpenAI 兼容 API**（支持 DeepSeek、Kimi、智谱等）
- **ChromaDB** 用于 RAG 知识库检索
- **uv** 包管理器

## RAG（检索增强生成）

RAG = VectorDB。核心思路很简单：
1. 把文档切成小块（chunk）
2. 用 embedding 模型把每个 chunk 转成向量
3. 存入向量数据库
4. 用户提问时，用同样的 embedding 模型把问题转成向量
5. 在数据库中搜索最相似的 chunks
6. 把搜索结果作为上下文，随问题一起发给 LLM

这个项目的 RAG 引擎代码在 `core/rag.py`，不到 200 行。
