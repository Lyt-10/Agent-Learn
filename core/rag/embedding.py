# core/rag/embedding.py
"""Embedding 适配器：把 OpenAI 兼容的 embedding API 包成 Chroma 认得的 EmbeddingFunction。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

BATCH_SIZE = 64      # 实测：智谱 embedding-2 单请求上限 64 条，超出报错 1214
MAX_CHARS = 800      # 实测：单条上限 512 token ≈ 800 汉字，超出会被静默截断

def load_env()->None:
    """加载 .env 文件中的环境变量。"""
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"未找到 .env 文件，请在项目根目录下创建 {ENV_PATH} 文件。")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

class EmbeddingClient:
    """一个极薄的 embedding API 客户端：只负责分批调用 + 溢出截断。"""
    def __init__(self)->None:
        load_env()
        api_key = os.environ.get("EMBEDDING_API_KEY")
        if not api_key:
            raise ValueError("未在 .env 文件中找到 EMBEDDING_API_KEY，请设置该环境变量。")
        self.model = os.environ.get("EMBEDDING_MODEL_ID", "embedding-2")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("EMBEDDING_BASE_URL"),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把任意条文本转成向量，自动按 BATCH_SIZE 分批。"""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = [text[:MAX_CHARS] for text in texts[start:start + BATCH_SIZE]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors

class ZhipuEmbeddingFunction(EmbeddingFunction[Documents]):
    """供 Chroma 使用的 embedding function。"""

    def __init__(self, batch_size: int = BATCH_SIZE, max_chars: int = MAX_CHARS) -> None:
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._client = EmbeddingClient()

    def __call__(self, input: Documents) -> Embeddings:
        texts = [str(text)[: self.max_chars] for text in input]
        # chromadb 把 Embeddings 标成 List[NDArray[...]]（类型过窄），而运行时实际接受的是
        # PyEmbedding = Union[Sequence[float], Sequence[int]]，纯 Python list 完全合法。
        return cast(Embeddings, self._client.embed(texts))

     # —— 下面四个方法是 Chroma 的协议要求，缺了重开集合时会退化 ——
    @staticmethod
    def name() -> str:
        return "zhipu-embedding-api"

    def get_config(self) -> dict:
        return {"model": self._client.model, "batch_size": self.batch_size}

    @staticmethod
    def build_from_config(config: dict) -> "ZhipuEmbeddingFunction":
        return ZhipuEmbeddingFunction(
            batch_size=int(config.get("batch_size", BATCH_SIZE)),
        )


@lru_cache(maxsize=1)
def get_embedding_function() -> ZhipuEmbeddingFunction:
    """进程内单例：避免每次建 collection 都新建客户端。"""
    return ZhipuEmbeddingFunction()


if __name__ == "__main__":
    ef = get_embedding_function()
    vecs = ef(["什么是 Node？", "Memory 怎么压缩上下文？"])
    print("条数:", len(vecs), "维度:", len(vecs[0]))