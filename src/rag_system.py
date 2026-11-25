"""RAGSystem: recuperador híbrido (BM25 + vectorial Pinecone) con EnsembleRetriever."""
import json
import os
import re
import warnings

warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*", category=DeprecationWarning)

from langchain_classic.retrievers import EnsembleRetriever  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from pinecone import Pinecone  # noqa: E402

from src import config  # noqa: E402
from src.vectorstore import get_vectorstore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())   # 'Depends(' == 'depends'


def load_corpus_from_pinecone(use_cache: bool = True) -> list[Document]:
    """Reconstruye el corpus léxico desde la metadata de Pinecone (sin otra BD)."""
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(config.INDEX_NAME)
    if use_cache and config.CORPUS_CACHE.exists():
        rows = [json.loads(l) for l in config.CORPUS_CACHE.read_text(encoding="utf-8").splitlines()]
        remote = index.describe_index_stats().namespaces.get(config.NAMESPACE)
        remote_n = remote.vector_count if remote else 0
        if len(rows) == remote_n:                 # caché válida solo si coincide con Pinecone
            return [Document(page_content=r.pop("text"), metadata=r) for r in rows]
        print(f"[cache] obsoleta ({len(rows)} vs {remote_n} en Pinecone): reconstruyendo")

    ids = [i for page in index.list(namespace=config.NAMESPACE) for i in page]
    docs = []
    for start in range(0, len(ids), 100):
        res = index.fetch(ids=ids[start:start + 100], namespace=config.NAMESPACE)
        for v in res.vectors.values():
            meta = dict(v.metadata)
            docs.append(Document(page_content=meta.pop("text"), metadata=meta))
    config.CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with config.CORPUS_CACHE.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({**d.metadata, "text": d.page_content}, ensure_ascii=False) + "\n")
    return docs


class RAGSystem:
    """Recuperador híbrido. `corpus` y `vectorstore` son inyectables (útil para tests sin red)."""

    def __init__(self, k: int = config.TOP_K, weights=(0.5, 0.5), use_cache: bool = True,
                 corpus: list[Document] | None = None, vectorstore=None):
        self.k = k
        self.corpus = corpus if corpus is not None else load_corpus_from_pinecone(use_cache)
        if not self.corpus:
            raise RuntimeError("Namespace vacío: ejecutá primero `python -m src.ingest`.")
        self.vectorstore = vectorstore if vectorstore is not None else get_vectorstore()
        self.bm25, self.semantic = self._build_retrievers(None)
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.semantic], weights=list(weights), id_key="chunk_id"
        )
        self._by_category: dict[str, EnsembleRetriever] = {}

    @property
    def categories(self) -> list[str]:
        return sorted({d.metadata.get("category", "general") for d in self.corpus})

    def _build_retrievers(self, category: str | None):
        """BM25 sobre el subcorpus de la categoría + retriever vectorial con filtro de metadata.
        Cada retriever trae 2k candidatos para que la fusión (RRF) tenga de dónde elegir."""
        docs = self.corpus if category is None else [
            d for d in self.corpus if d.metadata.get("category") == category]
        if not docs:
            raise ValueError(f"Categoría desconocida: '{category}'. Disponibles: {self.categories}")
        kwargs = {"k": 2 * self.k}
        if category is not None:
            kwargs["filter"] = {"category": {"$eq": category}}
        bm25 = BM25Retriever.from_documents(docs, k=2 * self.k, preprocess_func=_tokenize)
        return bm25, self.vectorstore.as_retriever(search_kwargs=kwargs)

    def _ensemble_for(self, category: str | None) -> EnsembleRetriever:
        if category is None:
            return self.ensemble
        if category not in self._by_category:
            self._by_category[category] = EnsembleRetriever(
                retrievers=list(self._build_retrievers(category)),
                weights=list(self.ensemble.weights), id_key="chunk_id",
            )
        ens = self._by_category[category]
        ens.weights = list(self.ensemble.weights)          # respeta cambios de pesos
        return ens

    def retrieve(self, query: str, category: str | None = None) -> list[Document]:
        """Top-k documentos combinando resultados léxicos y semánticos.
        Con `category`, ambos retrievers se restringen a esa categoría."""
        return self._ensemble_for(category).invoke(query)[: self.k]

    def retrieve_semantic(self, query: str, category: str | None = None) -> list[Document]:
        if category is None:
            return self.semantic.invoke(query)[: self.k]
        return self._build_retrievers(category)[1].invoke(query)[: self.k]


if __name__ == "__main__":
    rag = RAGSystem()
    for q, cat in [("BackgroundTasks", None), ("How do I declare optional query parameters?", "tutorial"),
                   ("lifespan events", "advanced")]:
        print(f"\n> {q}  [categoría={cat}]")
        for d in rag.retrieve(q, category=cat):
            print("  ", d.metadata["chunk_id"], "|", d.page_content[:60].replace("\n", " "))

