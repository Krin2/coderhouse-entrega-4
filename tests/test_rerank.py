import json

import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import DEFAULT_MODEL, PineconeReranker, RerankQuotaError


def doc(chunk_id, text="texto"):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "doc_id": chunk_id.split("#")[0],
                                                  "category": "tutorial"})


class Item:
    def __init__(self, index, score): self.index, self.score = index, score


class FakeInference:
    """Puntúa más alto a los documentos cuyo texto contiene 'relevante'; permite inyectar fallos."""

    def __init__(self, failures=()):
        self.calls, self.failures = [], list(failures)

    def rerank(self, **kw):
        self.calls.append(kw)
        if self.failures:
            raise self.failures.pop(0)
        scores = [1.0 if "relevante" in d["text"] else 0.1 * i for i, d in enumerate(kw["documents"])]
        return type("R", (), {"data": [Item(i, s) for i, s in enumerate(scores)]})()


class FakeClient:
    def __init__(self, **kw): self.inference = FakeInference(**kw)


class HttpError(Exception):
    def __init__(self, status, message=""):
        super().__init__(message)
        self.status = status


def make(client=None, **kw):
    kw.setdefault("sleep", lambda s: None)
    return PineconeReranker(client or FakeClient(), **kw)


DOCS = [doc("a#0", "otra cosa"), doc("b#0", "texto relevante"), doc("c#0", "mas texto")]


def test_ordena_por_puntaje_y_respeta_top_n():
    out = make().rerank("q", DOCS, top_n=2)
    assert [d.metadata["chunk_id"] for d in out] == ["b#0", "c#0"]


def test_envia_ids_texto_y_parametros_esperados():
    client = FakeClient()
    make(client).rerank("mi consulta", DOCS, top_n=2)
    call = client.inference.calls[0]
    assert call["model"] == DEFAULT_MODEL and call["query"] == "mi consulta"
    assert call["documents"][1] == {"id": "b#0", "text": "texto relevante"}
    assert call["rank_fields"] == ["text"] and call["return_documents"] is False
    assert call["parameters"] == {"truncate": "END"}


def test_sin_candidatos_no_consume_solicitudes():
    client = FakeClient()
    assert make(client).rerank("q", [], top_n=5) == []
    assert client.inference.calls == []


def test_cache_en_memoria_evita_la_segunda_solicitud():
    client = FakeClient()
    rr = make(client)
    rr.rerank("q", DOCS, 2)
    rr.rerank("q", DOCS, 2)
    assert len(client.inference.calls) == 1 and rr.requests == 1 and rr.cache_hits == 1


def test_la_cache_distingue_consulta_candidatos_y_modelo():
    client = FakeClient()
    rr = make(client)
    rr.rerank("q1", DOCS, 2)
    rr.rerank("q2", DOCS, 2)
    rr.rerank("q1", DOCS[:2], 2)
    assert rr.requests == 3
    other = make(client, model="otro-modelo")
    other.rerank("q1", DOCS, 2)
    assert other.requests == 1


def test_cache_en_disco_sobrevive_entre_instancias(tmp_path):
    path = tmp_path / "cache" / "rerank.jsonl"
    first = make(FakeClient(), cache_path=path)
    expected = first.rerank("q", DOCS, 3)
    client2 = FakeClient()
    second = make(client2, cache_path=path)
    assert second.rerank("q", DOCS, 3) == expected
    assert client2.inference.calls == [] and second.requests == 0 and second.cache_hits == 1
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["order"] == [1, 2, 0]


def test_ritmo_maximo_entre_solicitudes():
    now, waits = [100.0], []
    rr = make(clock=lambda: now[0], sleep=waits.append, min_interval=1.1)
    rr.rerank("q1", DOCS, 2)
    now[0] += 0.3                                   # solo pasaron 0.3 s desde la anterior
    rr.rerank("q2", DOCS, 2)
    assert waits and waits[-1] == pytest.approx(0.8)


def test_reintenta_el_limite_por_minuto_con_backoff():
    waits = []
    client = FakeClient(failures=[HttpError(429, "too many requests"), HttpError(429, "too many requests")])
    rr = make(client, sleep=waits.append, min_interval=0)
    assert rr.rerank("q", DOCS, 1)[0].metadata["chunk_id"] == "b#0"
    assert rr.requests == 3 and 1.0 in waits and 2.0 in waits          # backoff 1 s, 2 s


def test_cuota_mensual_no_se_reintenta():
    client = FakeClient(failures=[HttpError(429, "You've reached your read unit limit for the current month")])
    with pytest.raises(RerankQuotaError, match="Cuota mensual"):
        make(client).rerank("q", DOCS, 2)
    assert len(client.inference.calls) == 1


def test_429_persistente_termina_con_el_error_original():
    client = FakeClient(failures=[HttpError(429, "rate")] * 10)
    with pytest.raises(HttpError):
        make(client, max_retries=2).rerank("q", DOCS, 2)
    assert len(client.inference.calls) == 3


def test_otros_errores_se_propagan_sin_reintentar():
    client = FakeClient(failures=[HttpError(400, "bad request")])
    with pytest.raises(HttpError):
        make(client).rerank("q", DOCS, 2)
    assert len(client.inference.calls) == 1


def test_tope_de_solicitudes_corta_antes_de_llamar_y_conserva_la_cache(tmp_path):
    client = FakeClient()
    rr = make(client, max_requests=2, cache_path=tmp_path / "c.jsonl")
    rr.rerank("q1", DOCS, 2)
    rr.rerank("q2", DOCS, 2)
    with pytest.raises(RerankQuotaError, match="tope de 2"):
        rr.rerank("q3", DOCS, 2)
    assert len(client.inference.calls) == 2
    rr.rerank("q1", DOCS, 2)                         # lo ya calculado sigue disponible
    stats = rr.stats()
    assert (stats["model"], stats["requests"], stats["cache_hits"]) == (DEFAULT_MODEL, 2, 1)


# ---------- integración con HybridRetriever ----------
class ReverseReranker:
    def __init__(self): self.calls = []

    def rerank(self, query, docs, top_n):
        self.calls.append((query, [d.metadata["chunk_id"] for d in docs]))
        return list(reversed(docs))[:top_n]


def test_retrieve_reranked_reordena_los_candidatos(corpus, fake_store):
    rr = ReverseReranker()
    rag = HybridRetriever(corpus, fake_store, k=2, reranker=rr)
    plain = rag.retrieve("lifespan")
    reranked = rag.retrieve("lifespan", rerank=True)
    assert len(reranked) == 2 and rr.calls[0][0] == "lifespan"
    candidates = rr.calls[0][1]
    assert len(candidates) >= len(plain)                      # reordena un conjunto más amplio que el top-k
    assert [d.metadata["chunk_id"] for d in reranked] == list(reversed(candidates))[:2]


def test_retrieve_semantic_reranked_usa_solo_candidatos_vectoriales(corpus, fake_store):
    rr = ReverseReranker()
    rag = HybridRetriever(corpus, fake_store, k=2, reranker=rr)
    rag.retrieve_semantic_reranked("x")
    assert rr.calls[0][1] == [d.metadata["chunk_id"] for d in corpus[:4]]


def test_rerank_respeta_la_categoria(corpus, fake_store):
    rr = ReverseReranker()
    rag = HybridRetriever(corpus, fake_store, k=2, reranker=rr)
    rag.retrieve("x", category="advanced", rerank=True)
    assert all(c.startswith("advanced/") for c in rr.calls[0][1])


def test_sin_reranker_falla_con_mensaje_claro(corpus, fake_store):
    rag = HybridRetriever(corpus, fake_store)
    with pytest.raises(RuntimeError, match="reranker"):
        rag.retrieve("x", rerank=True)
    assert rag.retrieve("x")                                  # sin rerank sigue funcionando
