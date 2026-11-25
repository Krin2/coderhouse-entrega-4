import pytest

from src.rag_system import RAGSystem
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import tokenize


def test_tokenize_normaliza_identificadores():
    assert tokenize("Depends(get_db) y WebSocketDisconnect") == ["depends", "get_db", "y", "websocketdisconnect"]


def test_top_k_sin_duplicados(corpus, fake_store):
    res = HybridRetriever(corpus, fake_store, k=3).retrieve("BackgroundTasks")
    ids = [d.metadata["chunk_id"] for d in res]
    assert len(res) <= 3 and len(ids) == len(set(ids))


def test_bm25_aporta_termino_exacto(corpus, fake_store):
    # el vectorstore falso no entiende la consulta: el acierto solo puede venir de BM25
    top = HybridRetriever(corpus, fake_store, k=2).retrieve("WebSocketDisconnect")[0]
    assert top.metadata["chunk_id"] == "advanced/ws#0"


def test_retrieve_lexical_y_semantico(corpus, fake_store):
    rag = HybridRetriever(corpus, fake_store, k=2)
    assert rag.retrieve_lexical("lifespan")[0].metadata["chunk_id"] == "advanced/events#0"
    assert len(rag.retrieve_semantic("lo que sea")) == 2


def test_filtro_categoria_en_ambos_retrievers(corpus, fake_store):
    rag = HybridRetriever(corpus, fake_store, k=4)
    res = rag.retrieve("BackgroundTasks", category="advanced")
    assert res and {d.metadata["category"] for d in res} == {"advanced"}
    assert {"$eq": "advanced"} in [c["filter"]["category"] for c in fake_store.calls if "filter" in c]


def test_categoria_desconocida(corpus, fake_store):
    with pytest.raises(ValueError, match="Categoría desconocida"):
        HybridRetriever(corpus, fake_store).retrieve("x", category="inexistente")


def test_categories(corpus, fake_store):
    assert HybridRetriever(corpus, fake_store).categories == ["advanced", "tutorial"]


def test_corpus_vacio(fake_store):
    with pytest.raises(RuntimeError, match="Corpus vac"):
        HybridRetriever([], fake_store)


def test_pesos_se_propagan_a_la_categoria(corpus, fake_store):
    rag = HybridRetriever(corpus, fake_store)
    rag.set_weights((0.9, 0.1))
    rag.retrieve("lifespan", category="advanced")
    assert rag._ensembles["advanced"].weights == [0.9, 0.1]


def test_retrievers_por_categoria_se_cachean(corpus, fake_store):
    rag = HybridRetriever(corpus, fake_store)
    rag.retrieve("a", category="advanced")
    n = len(fake_store.calls)
    rag.retrieve("b", category="advanced")
    assert len(fake_store.calls) == n


def test_ragsystem_acepta_inyeccion_sin_credenciales(corpus, fake_store):
    rag = RAGSystem(corpus=corpus, vectorstore=fake_store, k=2)
    assert rag.k == 2 and len(rag.retrieve("BackgroundTasks")) <= 2


@pytest.mark.integration
def test_integracion_filtro_real():
    rag = RAGSystem()
    res = rag.retrieve("lifespan events", category="advanced")
    assert res and all(d.metadata["category"] == "advanced" for d in res)
