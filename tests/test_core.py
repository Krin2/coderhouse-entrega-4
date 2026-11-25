import json

import pytest

import evaluate
from src import config, ingest
from src.rag_system import RAGSystem, _tokenize, load_corpus_from_pinecone


# ---------- métricas ----------
def test_score_hit_en_top_k():
    assert evaluate.score(["a", "x", "a", "y", "z"], "a", 5) == (1.0, 0.4)


def test_score_sin_aciertos():
    assert evaluate.score(["x", "y"], "a", 5) == (0.0, 0.0)


def test_score_respeta_k():
    # el acierto está en la posición 4: con k=3 no cuenta
    assert evaluate.score(["x", "y", "z", "a"], "a", 3) == (0.0, 0.0)


def test_score_todos_relevantes():
    assert evaluate.score(["a"] * 5, "a", 5) == (1.0, 1.0)


def test_evaluate_promedia():
    class D:
        def __init__(self, i): self.metadata = {"doc_id": i}
    golden = [{"pregunta": "q1", "documento_id_esperado": "a"}, {"pregunta": "q2", "documento_id_esperado": "b"}]
    res = evaluate.evaluate(lambda q: [D("a")] * 2 + [D("z")] * 3 if q == "q1" else [D("z")] * 5, 5, golden)
    assert res["recall"] == 0.5 and res["precision"] == pytest.approx(0.2)


def test_golden_set_valido():
    golden = json.loads(evaluate.GOLDEN.read_text(encoding="utf-8-sig"))
    assert len(golden) >= 5
    assert all(set(g) == {"pregunta", "documento_id_esperado"} for g in golden)
    assert len({g["pregunta"] for g in golden}) == len(golden)


# ---------- tokenización y chunking ----------
def test_tokenize_normaliza_identificadores():
    assert _tokenize("Depends(get_db) y WebSocketDisconnect") == ["depends", "get_db", "y", "websocketdisconnect"]


def test_load_documents_metadata(tmp_path):
    (tmp_path / "tutorial").mkdir()
    (tmp_path / "tutorial" / "a.md").write_text("# A\n{* ref.py *}\ntexto", encoding="utf-8")
    (tmp_path / "top.md").write_text("# Top", encoding="utf-8")
    docs = {d.metadata["doc_id"]: d for d in ingest.load_documents(tmp_path)}
    assert docs["tutorial/a"].metadata["category"] == "tutorial"
    assert docs["top"].metadata["category"] == "general"
    assert "{*" not in docs["tutorial/a"].page_content


def test_split_documents_ids_y_secciones(tmp_path):
    body = "# Titulo { #titulo }\n\n" + "palabra " * 700 + "\n\n## Segunda { #segunda }\n\n" + "otra " * 700
    (tmp_path / "doc.md").write_text(body, encoding="utf-8")
    chunks = ingest.split_documents(ingest.load_documents(tmp_path))
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata["chunk_id"] == "doc#0"
    assert chunks[0].metadata["section"] == "Titulo"          # ancla { #x } removida
    assert chunks[-1].metadata["section"] == "Segunda"
    assert all(c.metadata["category"] == "general" for c in chunks)


# ---------- caché BM25 ----------
class _FakeIndex:
    def __init__(self, n):
        self.n = n
        self.fetched = False

    def describe_index_stats(self):
        class NS: vector_count = self.n
        class S: namespaces = {config.NAMESPACE: NS()}
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
def fake_pinecone(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CORPUS_CACHE", tmp_path / "corpus.jsonl")
    holder = {}

    class FakePC:
        def __init__(self, **kw): pass
        def Index(self, name): return holder["index"]

    monkeypatch.setattr("src.rag_system.Pinecone", FakePC)
    monkeypatch.setenv("PINECONE_API_KEY", "x")
    return holder


def test_cache_valida_no_consulta_pinecone(fake_pinecone):
    config.CORPUS_CACHE.write_text(
        '{"chunk_id": "a#0", "text": "t1"}\n{"chunk_id": "b#0", "text": "t2"}\n', encoding="utf-8")
    fake_pinecone["index"] = idx = _FakeIndex(2)
    docs = load_corpus_from_pinecone()
    assert [d.page_content for d in docs] == ["t1", "t2"] and not idx.fetched


def test_cache_obsoleta_se_reconstruye(fake_pinecone, capsys):
    config.CORPUS_CACHE.write_text('{"chunk_id": "a#0", "text": "viejo"}\n', encoding="utf-8")
    fake_pinecone["index"] = idx = _FakeIndex(2)          # Pinecone tiene 2, la caché 1
    docs = load_corpus_from_pinecone()
    assert idx.fetched and len(docs) == 2
    assert "obsoleta" in capsys.readouterr().out
    assert len(config.CORPUS_CACHE.read_text(encoding="utf-8").splitlines()) == 2   # caché reescrita


def test_sin_cache_consulta_pinecone(fake_pinecone):
    fake_pinecone["index"] = idx = _FakeIndex(2)
    assert len(load_corpus_from_pinecone()) == 2 and idx.fetched


# ---------- RAGSystem ----------
def test_ragsystem_top_k_sin_duplicados(corpus, fake_store):
    rag = RAGSystem(k=3, corpus=corpus, vectorstore=fake_store)
    res = rag.retrieve("BackgroundTasks")
    ids = [d.metadata["chunk_id"] for d in res]
    assert len(res) <= 3 and len(ids) == len(set(ids))


def test_bm25_aporta_termino_exacto(corpus, fake_store):
    rag = RAGSystem(k=2, corpus=corpus, vectorstore=fake_store)    # el "vectorial" falso no entiende la consulta
    assert rag.retrieve("WebSocketDisconnect")[0].metadata["chunk_id"] == "advanced/ws#0"


def test_filtro_categoria_en_ambos_retrievers(corpus, fake_store):
    rag = RAGSystem(k=4, corpus=corpus, vectorstore=fake_store)
    res = rag.retrieve("BackgroundTasks", category="advanced")
    assert res and {d.metadata["category"] for d in res} == {"advanced"}
    assert {"$eq": "advanced"} in [c["filter"]["category"] for c in fake_store.calls if "filter" in c]


def test_categoria_desconocida(corpus, fake_store):
    rag = RAGSystem(corpus=corpus, vectorstore=fake_store)
    with pytest.raises(ValueError, match="Categoría desconocida"):
        rag.retrieve("x", category="inexistente")


def test_categories(corpus, fake_store):
    assert RAGSystem(corpus=corpus, vectorstore=fake_store).categories == ["advanced", "tutorial"]


def test_corpus_vacio():
    with pytest.raises(RuntimeError, match="vac"):
        RAGSystem(corpus=[], vectorstore=object())


def test_pesos_se_propagan_a_la_categoria(corpus, fake_store):
    rag = RAGSystem(corpus=corpus, vectorstore=fake_store)
    rag.ensemble.weights = [0.9, 0.1]
    rag.retrieve("lifespan", category="advanced")
    assert rag._by_category["advanced"].weights == [0.9, 0.1]


# ---------- integración (opt-in) ----------
@pytest.mark.integration
def test_integracion_filtro_real():
    rag = RAGSystem()
    res = rag.retrieve("lifespan events", category="advanced")
    assert res and all(d.metadata["category"] == "advanced" for d in res)
