"""Fuente del corpus léxico (BM25): se reconstruye desde la metadata de Pinecone, con caché local."""
from __future__ import annotations

import json
from typing import Protocol

from langchain_core.documents import Document

from src.settings import Settings


class CorpusSource(Protocol):
    def load(self) -> list[Document]: ...


class PineconeCorpus:
    """Lee todos los chunks del namespace. La caché solo es válida si su tamaño coincide con Pinecone."""

    def __init__(self, settings: Settings, index, use_cache: bool = True, batch_size: int = 100):
        self.settings, self.index, self.use_cache, self.batch_size = settings, index, use_cache, batch_size

    def load(self) -> list[Document]:
        cached = self._read_cache()
        if cached is not None:
            remote_n = self._remote_count()
            if len(cached) == remote_n:
                return cached
            print(f"[cache] obsoleta ({len(cached)} vs {remote_n} en Pinecone): reconstruyendo")
        docs = self._fetch_all()
        self._write_cache(docs)
        return docs

    def _remote_count(self) -> int:
        ns = self.index.describe_index_stats().namespaces.get(self.settings.namespace)
        return ns.vector_count if ns else 0

    def _fetch_all(self) -> list[Document]:
        namespace = self.settings.namespace
        ids = [i for page in self.index.list(namespace=namespace) for i in page]
        docs = []
        for start in range(0, len(ids), self.batch_size):
            res = self.index.fetch(ids=ids[start:start + self.batch_size], namespace=namespace)
            for vector in res.vectors.values():
                meta = dict(vector.metadata)
                docs.append(Document(page_content=meta.pop("text"), metadata=meta))
        return docs

    def _read_cache(self) -> list[Document] | None:
        path = self.settings.corpus_cache
        if not (self.use_cache and path.exists()):
            return None
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [Document(page_content=row.pop("text"), metadata=row) for row in rows]

    def _write_cache(self, docs: list[Document]) -> None:
        path = self.settings.corpus_cache
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({**d.metadata, "text": d.page_content}, ensure_ascii=False) + "\n")
