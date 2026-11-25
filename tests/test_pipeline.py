import pytest

from src.ingestion.pipeline import IngestPipeline
from src.settings import Settings


class FakeLoader:
    def __init__(self, docs): self.docs = docs
    def load(self): return self.docs


class FakeStore:
    def __init__(self): self.added = None
    def add_documents(self, chunks, ids, batch_size): self.added = (chunks, ids, batch_size)


class FakeIndex:
    def __init__(self, error=None): self.error, self.deleted = error, None
    def delete(self, delete_all, namespace):
        if self.error: raise self.error
        self.deleted = (delete_all, namespace)


def _pipeline(tmp_path, docs, index=None):
    settings = Settings(namespace="ns", cache_dir=tmp_path)
    return IngestPipeline(settings, FakeLoader(docs), FakeStore(), index or FakeIndex()), settings


def _docs():
    from langchain_core.documents import Document
    return [Document(page_content="# T\n\ntexto", metadata={"doc_id": "d", "category": "c", "source": "d.md"})]


def test_run_sube_chunks_con_ids_deterministicos(tmp_path):
    pipe, _ = _pipeline(tmp_path, _docs())
    assert pipe.run() == 1
    chunks, ids, batch = pipe.vectorstore.added
    assert ids == ["d#0"] and chunks[0].metadata["chunk_id"] == "d#0" and batch == 100


def test_run_resetea_namespace_e_invalida_cache(tmp_path):
    pipe, settings = _pipeline(tmp_path, _docs())
    settings.corpus_cache.write_text("viejo", encoding="utf-8")
    pipe.run()
    assert pipe.index.deleted == (True, "ns") and not settings.corpus_cache.exists()


def test_run_sin_documentos_falla(tmp_path):
    pipe, _ = _pipeline(tmp_path, [])
    with pytest.raises(RuntimeError, match="No hay documentos"):
        pipe.run()
    assert pipe.vectorstore.added is None


def test_reset_tolera_namespace_inexistente(tmp_path):
    pipe, _ = _pipeline(tmp_path, _docs(), FakeIndex(Exception("Namespace not found")))
    pipe.reset_namespace()                                    # 1a ingesta: no debe fallar


def test_reset_propaga_otros_errores(tmp_path):
    pipe, _ = _pipeline(tmp_path, _docs(), FakeIndex(Exception("unauthorized")))
    with pytest.raises(Exception, match="unauthorized"):
        pipe.reset_namespace()
