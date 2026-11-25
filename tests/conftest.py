"""Fixtures: corpus sintético y vectorstore falso (sin red)."""
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_doc(chunk_id: str, text: str, category: str) -> Document:
    return Document(page_content=text, metadata={
        "chunk_id": chunk_id, "doc_id": chunk_id.split("#")[0], "category": category, "source": f"{chunk_id}.md"})


@pytest.fixture
def corpus():
    return [
        make_doc("tutorial/bg#0", "Use BackgroundTasks to run work after returning a response", "tutorial"),
        make_doc("tutorial/cors#0", "CORSMiddleware allow_origins configuration for browsers", "tutorial"),
        make_doc("advanced/ws#0", "WebSocketDisconnect is raised when the client disconnects", "advanced"),
        make_doc("advanced/events#0", "lifespan startup and shutdown events", "advanced"),
    ]


class FakeVectorStore:
    """Imita as_retriever(): devuelve docs del corpus filtrando por metadata, en orden fijo."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.calls = []

    def as_retriever(self, search_kwargs=None):
        store, kwargs = self, search_kwargs or {}
        store.calls.append(kwargs)

        class _R(BaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                cat = kwargs.get("filter", {}).get("category", {}).get("$eq")
                docs = [d for d in store.corpus if cat is None or d.metadata["category"] == cat]
                return docs[: kwargs.get("k", 4)]

        return _R()


@pytest.fixture
def fake_store(corpus):
    return FakeVectorStore(corpus)
