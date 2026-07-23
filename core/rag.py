"""RAG Engine - ChromaDB + OpenAI-compatible embeddings.

极简 RAG 引擎，遵循项目理念 "RAG = VectorDB"。

使用示例:
    from core.rag import RAGEngine

    engine = RAGEngine(collection_name="my_docs")
    engine.index_file("docs/readme.md")
    engine.index_directory("docs/")
    print(engine.query("什么是 Agent?"))
"""

from __future__ import annotations

import fnmatch
import os
import uuid
from pathlib import Path
from typing import Any

# 自动加载项目根目录的 .env 文件
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            if _key.strip() not in os.environ:
                os.environ[_key.strip()] = _val.strip()

import chromadb
from chromadb.utils import embedding_functions


class RAGEngine:
    """ChromaDB-backed vector store with OpenAI-compatible embeddings."""

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ) -> None:
        # ── 解析持久化目录 ──
        if persist_dir is None:
            persist_dir = os.environ.get("CHROMA_PERSIST_DIR")
        if persist_dir is None:
            persist_dir = str(Path.cwd() / "rag_data" / "chroma_db")

        # ── 解析 embedding 配置（优先级：参数 > 专用环境变量 > LLM 环境变量） ──
        api_key = (
            embedding_api_key
            or os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = (
            embedding_base_url
            or os.environ.get("EMBEDDING_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        model = embedding_model or os.environ.get("EMBEDDING_MODEL_ID")

        if not model:
            raise ValueError(
                "❌ EMBEDDING_MODEL_ID 未设置。请在 .env 中配置 embedding 模型。\n"
                "   示例: EMBEDDING_MODEL_ID=text-embedding-3-small\n"
                "   DeepSeek 不支持 embedding API，推荐使用 Kimi 或智谱的 embedding 服务。\n"
                "   完整 .env 配置示例:\n"
                "   EMBEDDING_API_KEY=sk-xxxx\n"
                "   EMBEDDING_BASE_URL=https://api.moonshot.cn/v1\n"
                "   EMBEDDING_MODEL_ID=kimi-embedding"
            )

        # ── 创建 ChromaDB 客户端 ──
        self._embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            api_base=base_url,
            model_name=model,
        )
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

    # ── 索引方法 ──

    def index_text(self, text: str, metadata: dict | None = None, doc_id: str | None = None) -> str:
        """索引一段文本。返回分配的 doc_id。

        Args:
            text: 要索引的文本。
            metadata: 元数据（如 {"source": "file.md", "chunk_index": 0}）。
            doc_id: 文档 ID，不传则自动生成 UUID。

        Returns:
            文档 ID 字符串。
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        self._collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        return doc_id

    def index_file(self, file_path: str, chunk_size: int | None = None) -> list[str]:
        """读取文件并索引。返回所有 chunk 的 doc_id 列表。

        Args:
            file_path: 文件路径。
            chunk_size: 分块大小（字符数），默认 2000。

        Returns:
            文档 ID 列表。
        """
        path = Path(file_path)
        if not path.exists():
            print(f"  [RAG] ⚠️ 文件不存在: {file_path}")
            return []

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"  [RAG] ⚠️ 读取文件失败: {file_path} — {e}")
            return []

        if not content.strip():
            print(f"  [RAG] ⚠️ 文件为空: {file_path}")
            return []

        chunks = self._chunk_content(content, path.suffix.lower(), chunk_size)

        doc_ids = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{path.name}:{i}"
            self.index_text(
                text=chunk,
                metadata={"source": str(path), "chunk_index": i},
                doc_id=doc_id,
            )
            doc_ids.append(doc_id)

        return doc_ids

    def index_directory(
        self,
        dir_path: str,
        glob_pattern: str = "*.{txt,md}",
        chunk_size: int | None = None,
    ) -> dict[str, list[str]]:
        """索引目录中所有匹配的文件。返回 {file_path: [doc_ids]}。

        Args:
            dir_path: 目录路径。
            glob_pattern: 文件匹配模式，默认 "*.{txt,md}"。
            chunk_size: 分块大小。

        Returns:
            {文件路径: 文档 ID 列表} 的字典。
        """
        dir_path_obj = Path(dir_path)
        if not dir_path_obj.exists():
            print(f"  [RAG] ⚠️ 目录不存在: {dir_path}")
            return {}

        # 解析扩展名列表
        suffixes = _parse_suffixes(glob_pattern)

        result: dict[str, list[str]] = {}
        for file_path in sorted(dir_path_obj.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in suffixes:
                doc_ids = self.index_file(str(file_path), chunk_size)
                if doc_ids:
                    result[str(file_path)] = doc_ids

        return result

    # ── 查询方法 ──

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜索知识库，返回结构化结果列表。

        Args:
            query: 搜索查询。
            top_k: 返回结果数，默认 5。

        Returns:
            结果列表，每项包含 id, text, metadata, distance。
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(query_texts=[query], n_results=min(top_k, self._collection.count()))

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items: list[dict[str, Any]] = []
        for i in range(len(ids)):
            items.append({
                "id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": distances[i] if i < len(distances) else 0.0,
            })

        return items

    def query(self, query: str, top_k: int = 5) -> str:
        """搜索并返回格式化的可读文本（供 Agent 工具使用）。

        Args:
            query: 搜索查询。
            top_k: 返回结果数。

        Returns:
            格式化的搜索结果字符串。
        """
        items = self.search(query, top_k)
        if not items:
            return "知识库中没有找到相关文档。"

        lines: list[str] = []
        for i, item in enumerate(items, 1):
            source = item["metadata"].get("source", "unknown")
            dist = item["distance"]
            text = item["text"][:300]
            lines.append(f"[{i}] (相关度: {dist:.3f}) 来源: {source}\n{text}")

        return "\n\n".join(lines)

    def count(self) -> int:
        """返回当前集合中的文档数量。"""
        return self._collection.count()

    def clear(self) -> None:
        """删除集合中的所有文档。"""
        ids = self._collection.get()["ids"]
        if ids:
            self._collection.delete(ids=ids)

    # ── 内部分块逻辑 ──

    def _chunk_content(self, content: str, suffix: str, chunk_size: int | None) -> list[str]:
        """根据文件类型选择分块策略。"""
        max_size = chunk_size if chunk_size else 2000

        if len(content) <= max_size:
            return [content]

        if suffix == ".md":
            return self._chunk_markdown(content, max_size)
        return self._chunk_text(content, max_size)

    @staticmethod
    def _chunk_markdown(content: str, max_size: int) -> list[str]:
        """按 ## 标题切分 Markdown，过长的节再按段落切分。"""
        # 按二级标题拆分
        sections: list[str] = []
        current = ""
        for line in content.splitlines(True):
            if line.startswith("## ") and current.strip():
                sections.append(current)
                current = line
            else:
                current += line
        if current.strip():
            sections.append(current)

        if not sections:
            sections = [content]

        # 过长的 section 按段落二次拆分
        chunks: list[str] = []
        for section in sections:
            if len(section) <= max_size:
                chunks.append(section)
            else:
                for sub in RAGEngine._split_by_paragraphs(section, max_size):
                    chunks.append(sub)

        return chunks or [content]

    @staticmethod
    def _chunk_text(content: str, max_size: int) -> list[str]:
        """按段落边界切分纯文本。"""
        return RAGEngine._split_by_paragraphs(content, max_size) or [content]

    @staticmethod
    def _split_by_paragraphs(content: str, max_size: int) -> list[str]:
        """通用按段落切分，超长段落按字符截断。"""
        paragraphs = content.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                # 单段超长时按字符截断
                if len(para) > max_size:
                    for i in range(0, len(para), max_size):
                        chunks.append(para[i:i + max_size])
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks


# ── 模块级工具函数 ──

def _parse_suffixes(glob_pattern: str) -> set[str]:
    """从 glob 模式中提取文件扩展名集合。

    >>> _parse_suffixes("*.{txt,md}")
    {'.txt', '.md'}
    >>> _parse_suffixes("*.py")
    {'.py'}
    """
    import re

    suffixes: set[str] = set()
    # 从模式中移除 * 号，提取扩展名
    # 处理 "*.{txt,md}" 这种 brace 模式
    brace_match = re.search(r"\{(.+?)\}", glob_pattern)
    if brace_match:
        for ext in brace_match.group(1).split(","):
            suffixes.add("." + ext.strip())
    else:
        # 假设模式是 "*.ext" 格式
        suffix = glob_pattern.lstrip("*")
        if suffix:
            suffixes.add(suffix)

    return suffixes
