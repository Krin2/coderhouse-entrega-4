"""Pipeline de ingesta: Loader -> chunking -> (reset del namespace) -> embeddings -> Pinecone."""
from __future__ import annotations

from src.infra.index import ensure_index, get_index
from src.ingestion.chunking import split_documents
from src.ingestion.loaders import Loader, default_loader
from src.settings import Settings
from src.vectorstore import get_vectorstore


class IngestPipeline:
    """Las dependencias (loader, vectorstore, index) son inyectables para poder testear sin red."""

    def __init__(self, settings: Settings, loader: Loader, vectorstore, index):
        self.settings, self.loader, self.vectorstore, self.index = settings, loader, vectorstore, index

    @classmethod
    def from_settings(cls, settings: Settings, with_code: bool = False) -> "IngestPipeline":
        """`with_code=True` incrusta el código referenciado por la documentación (requiere `settings.code_dir`)."""
        if with_code and not settings.code_dir.is_dir():
            raise FileNotFoundError(f"No existe {settings.code_dir}. Copiá docs_src de FastAPI ahí (ver README).")
        settings.require_keys()
        ensure_index(settings)
        loader = default_loader(settings.raw_dir, settings.code_dir if with_code else None)
        return cls(settings, loader, get_vectorstore(settings), get_index(settings))

    def reset_namespace(self) -> None:
        """Deja el namespace como espejo de los documentos fuente e invalida la caché BM25."""
        try:
            self.index.delete(delete_all=True, namespace=self.settings.namespace)
        except Exception as exc:                      # namespace inexistente en la 1a ingesta
            if "not found" not in str(exc).lower():
                raise
        self.settings.corpus_cache.unlink(missing_ok=True)

    def run(self) -> int:
        docs = self.loader.load()
        if not docs:
            raise RuntimeError(f"No hay documentos en {self.settings.raw_dir}. Ver README (descarga del dataset).")
        chunks = split_documents(
            docs, chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap,
            min_tokens=self.settings.min_chunk_tokens, max_tokens=self.settings.max_chunk_tokens)
        print(f"{len(docs)} documentos -> {len(chunks)} chunks")
        self.reset_namespace()
        # ids determinísticos -> reingestar es idempotente
        self.vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks], batch_size=100)
        print(f"[ok] {len(chunks)} chunks subidos al namespace '{self.settings.namespace}'")
        return len(chunks)
