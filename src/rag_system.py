"""RAGSystem: recuperador híbrido (BM25 + vectorial Pinecone) con EnsembleRetriever."""
import json
import os
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from pinecone import Pinecone

from src import config
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
    def __init__(self, k: int = config.TOP_K, weights=(0.5, 0.5), use_cache: bool = True):
        self.k = k
        corpus = load_corpus_from_pinecone(use_cache)
        if not corpus:
            raise RuntimeError("Namespace vacío: ejecutá primero `python -m src.ingest`.")
        # Cada retriever trae más candidatos (2k) para que la fusión (RRF) tenga de dónde elegir.
        self.bm25 = BM25Retriever.from_documents(corpus, k=2 * k, preprocess_func=_tokenize)
        self.semantic = get_vectorstore().as_retriever(search_kwargs={"k": 2 * k})
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25, self.semantic], weights=list(weights), id_key="chunk_id"
        )

    def retrieve(self, query: str) -> list[Document]:
        """Top-k documentos combinando resultados léxicos y semánticos."""
        return self.ensemble.invoke(query)[: self.k]

    def retrieve_semantic(self, query: str) -> list[Document]:
        return self.semantic.invoke(query)[: self.k]


if __name__ == "__main__":
    rag = RAGSystem()
    for q in ["BackgroundTasks", "How do I declare optional query parameters?"]:
        print(f"\n> {q}")
        for d in rag.retrieve(q):
            print("  ", d.metadata["chunk_id"], "|", d.page_content[:60].replace("\n", " "))

