import json

import pytest

from src.retrieval.corpus import PineconeCorpus
from src.settings import Settings


class FakeIndex:
    def __init__(self, n_remote, ns="ns"):
        self.n, self.ns, self.fetched = n_remote, ns, False

    def describe_index_stats(self):
        ns, n = self.ns, self.n
        class NS: vector_count = n
        class S: namespaces = {ns: NS()} if n else {}
        return S()

    def list(self, namespace=None):
        self.fetched = True
        yield ["a#0", "b#0"]

    def fetch(self, ids, namespace=None):
        class V:
            def __init__(s, i): s.metadata = {"text": f"texto {i}", "chunk_id": i}
        class R: vectors = {i: V(i) for i in ids}
        return R()


@pytest.fixture
def settings(tmp_path):
    return Settings(namespace="ns", cache_dir=tmp_path / "cache")


def _write_cache(settings, n):
    settings.corpus_cache.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"chunk_id": f"c{i}#0", "text": f"t{i}"}) for i in range(n)]
    settings.corpus_cache.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_cache_valida_no_descarga(settings):
    _write_cache(settings, 2)
    idx = FakeIndex(2)
    docs = PineconeCorpus(settings, idx).load()
    assert [d.page_content for d in docs] == ["t0", "t1"] and not idx.fetched


def test_cache_obsoleta_se_reconstruye(settings, capsys):
    _write_cache(settings, 1)                                 # Pinecone tiene 2
    idx = FakeIndex(2)
    docs = PineconeCorpus(settings, idx).load()
    assert idx.fetched and len(docs) == 2
    assert "obsoleta" in capsys.readouterr().out
    assert len(settings.corpus_cache.read_text(encoding="utf-8").splitlines()) == 2


def test_sin_cache_descarga_y_la_crea(settings):
    idx = FakeIndex(2)
    assert len(PineconeCorpus(settings, idx).load()) == 2 and idx.fetched
    assert settings.corpus_cache.exists()


def test_use_cache_false_ignora_la_cache(settings):
    _write_cache(settings, 2)
    idx = FakeIndex(2)
    PineconeCorpus(settings, idx, use_cache=False).load()
    assert idx.fetched


def test_namespace_vacio_en_pinecone_invalida_cache(settings):
    _write_cache(settings, 2)
    idx = FakeIndex(0)
    PineconeCorpus(settings, idx).load()
    assert idx.fetched
