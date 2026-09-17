# core/rag/index.py
"""离线索引 CLI：python -m core.rag.index <路径...> [--collection 名] [--rebuild] [--stats]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.rag.chunk import build_chunks, file_hash, iter_files
from core.rag.store import DEFAULT_COLLECTION, RagStore


def index_paths(
    paths: list[str],
    collection_name: str = DEFAULT_COLLECTION,
    rebuild: bool = False,
) -> dict:
    if rebuild:
        # 全量重建：连 collection 一起删掉（换 embedding 模型/维度时必须）
        RagStore(collection_name=collection_name).client.delete_collection(collection_name)

    store = RagStore(collection_name=collection_name)
    known = {} if rebuild else store.source_hashes()

    project_root = Path.cwd()
    seen: set[str] = set()
    added_files = skipped_files = added_chunks = 0

    for path in iter_files(paths):
        try:
            source = path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            source = path.resolve().as_posix()

        seen.add(source)
        content = path.read_text(encoding="utf-8", errors="ignore")
        digest = file_hash(content)

        if known.get(source) == digest:
            skipped_files += 1
            continue

        if source in known:
            store.delete_sources([source])                    # 变了：先清旧块

        chunks = build_chunks(path, source)
        added_chunks += store.add_chunks(chunks)
        added_files += 1
        print(f"  + {source} ({len(chunks)} 块)")

    removed = [source for source in known if source not in seen]
    store.delete_sources(removed)

    return {
        "added_files": added_files,
        "skipped_files": skipped_files,
        "removed_files": len(removed),
        "added_chunks": added_chunks,
        **store.stats(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="把文档索引进本地向量库")
    parser.add_argument("paths", nargs="*", default=["README.md", "learn.md", "core", "tools"],
                        help="要索引的文件或目录")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--rebuild", action="store_true", help="删除 collection 后全量重建")
    parser.add_argument("--stats", action="store_true", help="只打印统计信息")
    args = parser.parse_args(argv)

    if args.stats:
        print(RagStore(collection_name=args.collection).stats())
        return

    missing = [p for p in args.paths if not Path(p).exists()]
    if missing:
        print(f"路径不存在: {missing}")
        sys.exit(1)

    result = index_paths(args.paths, args.collection, args.rebuild)
    print(
        f"\n✅ 完成：新增/更新 {result['added_files']} 个文件、"
        f"{result['added_chunks']} 块；跳过 {result['skipped_files']} 个未变文件；"
        f"清理 {result['removed_files']} 个已删除文件\n"
        f"📦 库现状：{result['files']} 个文件 / {result['chunks']} 块 @ {result['persist_dir']}"
    )


if __name__ == "__main__":
    main()