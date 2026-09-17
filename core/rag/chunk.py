# core/rag/chunk.py
"""文本切片：把文件切成适合 embedding 的小块，并附上可回溯的 metadata。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

CHUNK_SIZE = 400        # 汉字数上限（≈250 token，远低于 512 上限）
CHUNK_OVERLAP = 60      # 相邻块重叠，避免答案被切断

TEXT_SUFFIXES = {".md", ".txt", ".py", ".ts", ".json", ".yaml", ".yml", ".toml"}
SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", "rag_data", "chat_memory", ".idea"}

@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict  # 只能是 str / int / float / bool，不能是 None 或嵌套结构

def make_chunk_id(source: str, chunk_index: int) -> str:
    """稳定 id：同一文件同一块，重复索引时 upsert 覆盖而不是新增。"""
    digest = hashlib.sha1(f"{source}:{chunk_index}".encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{chunk_index}"

def file_hash(text: str) -> str:
    """文件内容指纹，用于增量索引判断"要不要重新 embedding"。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def iter_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    """展开用户给的路径：目录递归、按后缀过滤、跳过黑名单目录。"""
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            yield path
            continue
        for item in sorted(path.rglob("*")):
            if item.is_file() and item.suffix.lower() in TEXT_SUFFIXES:
                if not SKIP_DIRS.intersection(item.parts):
                    yield item


def split_paragraphs(text: str) -> list[str]:
    """按空行切段落，段落内再按行兜底（代码文件常常没有空行）。"""
    blocks: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= CHUNK_SIZE:
            blocks.append(paragraph)
            continue
        # 超长段落（例如整段代码）按行累积切
        buffer = ""
        for line in paragraph.splitlines(keepends=True):
            if len(buffer) + len(line) > CHUNK_SIZE and buffer:
                blocks.append(buffer.strip())
                buffer = line
            else:
                buffer += line
        if buffer.strip():
            blocks.append(buffer.strip())
    return blocks


def pack_blocks(blocks: list[str]) -> list[str]:
    """把段落打包成 ≤CHUNK_SIZE 的块，并给每块加一点前一块的尾巴做重叠。"""
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-CHUNK_OVERLAP:]          # 重叠区
            current = f"{tail}\n\n{block}".strip()
        else:
            current = block
    if current:
        chunks.append(current)
    return chunks


def build_chunks(path: Path, source: str) -> list[Chunk]:
    """把一个文件切成 Chunk 列表。source 是相对路径，便于人读和引用。"""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    blocks = pack_blocks(split_paragraphs(text))
    digest = file_hash(text)
    mtime = path.stat().st_mtime
    source_key = Path(source).as_posix()

    chunks: list[Chunk] = []
    for index, block in enumerate(blocks):
        chunks.append(
            Chunk(
                id=make_chunk_id(source_key, index),
                text=block,
                metadata={
                    "source": source_key,
                    "chunk_index": index,
                    "mtime": float(mtime),
                    "hash": digest,
                },
            )
        )
    return chunks