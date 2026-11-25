"""Recuperador híbrido genérico: BM25 + vectorial combinados con EnsembleRetriever (RRF)."""
from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.retrieval.lexical import build_bm25


class HybridRetriever:
    """No conoce Pinecone ni OpenAI: recibe el corpus y cualquier vectorstore con `as_retriever()`."""

    def __init__(self, corpus: list[Document], vectorstore, *, k: int = 5, weights=(0.5, 0.5)):
        if not corpus:
            raise RuntimeError("Corpus vacío: ejecutá primero la ingesta (`python -m src.cli ingest`).")
        self.k, self.corpus, self.vectorstore = k, corpus, vectorstore
        self._parts: dict[str | None, tuple] = {}
        self._ensembles: dict[str | None, EnsembleRetriever] = {}
        self.bm25, self.semantic = self._parts_for(None)
        self.ensemble = self._ensembles[None] = EnsembleRetriever(
            retrievers=[self.bm25, self.semantic], weights=list(weights), id_key="chunk_id")

    # ---- API pública ----
    @property
    def categories(self) -> list[str]:
        return sorted({d.metadata.get("category", "general") for d in self.corpus})

    def set_weights(self, weights) -> None:
        self.ensemble.weights = list(weights)

    def retrieve(self, query: str, category: str | None = None) -> list[Document]:
        """Top-k combinando resultados léxicos y semánticos; `category` restringe ambos."""
        return self._ensemble_for(category).invoke(query)[: self.k]

    def retrieve_semantic(self, query: str, category: str | None = None) -> list[Document]:
        return self._parts_for(category)[1].invoke(query)[: self.k]

    def retrieve_lexical(self, query: str, category: str | None = None) -> list[Document]:
        return self._parts_for(category)[0].invoke(query)[: self.k]

    # ---- internos ----
    def _parts_for(self, category: str | None):
        """(BM25 sobre el subcorpus, retriever vectorial con filtro de metadata). Cada uno trae 2k
        candidatos para que la fusión tenga de dónde elegir."""
        if category not in self._parts:
            docs = self.corpus if category is None else [
                d for d in self.corpus if d.metadata.get("category") == category]
            if not docs:
                raise ValueError(f"Categoría desconocida: '{category}'. Disponibles: {self.categories}")
            kwargs = {"k": 2 * self.k}
            if category is not None:
                kwargs["filter"] = {"category": {"$eq": category}}
            self._parts[category] = (build_bm25(docs, 2 * self.k), self.vectorstore.as_retriever(search_kwargs=kwargs))
        return self._parts[category]

    def _ensemble_for(self, category: str | None) -> EnsembleRetriever:
        if category not in self._ensembles:
            self._ensembles[category] = EnsembleRetriever(
                retrievers=list(self._parts_for(category)), weights=list(self.ensemble.weights), id_key="chunk_id")
        ensemble = self._ensembles[category]
        ensemble.weights = list(self.ensemble.weights)          # respeta cambios de pesos
        return ensemble
