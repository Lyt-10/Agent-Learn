# core/rag/store.py
"""Chroma 封装：只管三件事——写入、检索、按文件维护。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction

from core.rag.chunk import Chunk
from core.rag.embedding import get_embedding_function

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "rag_data" / "chroma_db"
DEFAULT_COLLECTION = "agent_learn_kb"


def get_persist_dir() -> str:
    """库目录：优先 CHROMA_PERSIST_DIR，相对路径按项目根解析。"""
    raw = os.environ.get("CHROMA_PERSIST_DIR", "").strip()
    if not raw:
        return str(DEFAULT_PERSIST_DIR)
    path = Path(raw).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


@dataclass(slots=True)
class SearchHit:
    text: str
    source: str
    chunk_index: int
    score: float          # cosine 相似度：1 - chroma 返回的 distance

    def to_line(self) -> str:
        return (
            f"[source={self.source} chunk={self.chunk_index} score={self.score:.3f}]\n"
            f"{self.text}"
        )


class RagStore:
    """对 Chroma collection 的最小封装。"""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: str | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_dir or get_persist_dir())
        # ⚠️ embedding_function 必须每次显式传，否则 Chroma 会用自己的默认 ONNX 模型
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # chromadb 把形参标成 EmbeddingFunction[Embeddable]（含图片），而我们的 EF
            # 只处理 Documents。Protocol 在入参位置是逆变的，不加 cast 过不了类型检查。
            embedding_function=cast(
                EmbeddingFunction[Embeddable], get_embedding_function()
            ),
            metadata={"hnsw:space": "cosine"},      # 中文场景用余弦距离最稳
        )

    # ---------- 写入 ----------
    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )
        return len(chunks)

    def delete_sources(self, sources: list[str]) -> None:
        for source in sources:
            self.collection.delete(where={"source": source})

    def source_hashes(self) -> dict[str, str]:
        """{相对路径: 内容 hash}，增量索引据此判断哪些文件变了。"""
        if self.collection.count() == 0:
            return {}
        data = self.collection.get(include=["metadatas"])
        result: dict[str, str] = {}
        for metadata in data.get("metadatas") or []:
            if metadata and metadata.get("source"):
                result[str(metadata["source"])] = str(metadata.get("hash", ""))
        return result

    # ---------- 检索 ----------
    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        if self.collection.count() == 0 or not query.strip():
            return []

        # 带 where 过滤时多取一些候选，避免过滤后不够 top_k
        n_results = min(max(top_k * 3, top_k), self.collection.count())
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            # 别用 **{...} 解包：Pylance 无法判断 key 该落到哪个形参，会误报到 query_images 上。
            # 显式写 where=；chroma 内部 validate_filter_set 对 None 有判空，传 None 是安全的。
            where={"source": source} if source else None,
        )

        hits: list[SearchHit] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for text, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 - float(distance)
            if min_score is not None and score < min_score:
                continue
            # metadata 的值是联合类型（还可能是 SparseVector），int() 不接受；
            # 先 isinstance 收窄，顺便挡住脏数据把 int() 打崩。
            raw_index = (metadata or {}).get("chunk_index", -1)
            hits.append(
                SearchHit(
                    text=text,
                    source=str((metadata or {}).get("source", "unknown")),
                    chunk_index=raw_index if isinstance(raw_index, int) else -1,
                    score=score,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    # ---------- 统计 ----------
    def stats(self) -> dict[str, object]:
        sources = self.source_hashes()
        return {
            "persist_dir": get_persist_dir(),
            "collection": self.collection.name,
            "chunks": self.collection.count(),
            "files": len(sources),
        }