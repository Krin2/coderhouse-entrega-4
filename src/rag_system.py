"""RAGSystem: fachada que cablea Settings + Pinecone + OpenAI sobre el HybridRetriever."""
from __future__ import annotations

from langchain_core.documents import Document

from src.infra.index import get_index
from src.retrieval.corpus import PineconeCorpus
from src.retrieval.hybrid import HybridRetriever
from src.settings import Settings
from src.vectorstore import get_vectorstore


class RAGSystem(HybridRetriever):
    """Sistema listo para usar: `RAGSystem().retrieve("consulta", category="advanced")`.
    `corpus` y `vectorstore` se pueden inyectar (tests, otros backends)."""

    def __init__(self, settings: Settings | None = None, *, k: int | None = None, weights=(0.5, 0.5),
                 use_cache: bool = True, corpus: list[Document] | None = None, vectorstore=None):
        self.settings = settings or Settings.from_env()
        if corpus is None or vectorstore is None:
            self.settings.require_keys()
        if corpus is None:
            corpus = PineconeCorpus(self.settings, get_index(self.settings), use_cache).load()
        if vectorstore is None:
            vectorstore = get_vectorstore(self.settings)
        super().__init__(corpus, vectorstore, k=k or self.settings.top_k, weights=weights)
