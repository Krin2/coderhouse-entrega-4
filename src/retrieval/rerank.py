"""Reranking de candidatos con el servicio de inferencia de Pinecone (`bge-reranker-v2-m3` por defecto).

Cuida la cuota (plan Starter: 500 solicitudes/mes, 60/min): caché en disco, ritmo máximo y tope de solicitudes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Protocol

from langchain_core.documents import Document

DEFAULT_MODEL = "bge-reranker-v2-m3"       # 1024 tokens por par consulta-documento: aceptan chunks de hasta 800


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]: ...


class RerankQuotaError(RuntimeError):
    """Se agotó la cuota mensual del plan o el tope de solicitudes configurado: reintentar no sirve."""


class PineconeReranker:
    """`client` es un `Pinecone` (o un doble con `client.inference.rerank`). Una solicitud reordena todos
    los candidatos de una consulta. Los resultados se guardan en `cache_path` (JSONL) por
    (modelo, consulta, ids de chunk): repetir una evaluación no consume cuota."""

    def __init__(self, client, model: str = DEFAULT_MODEL, *, cache_path: Path | None = None,
                 min_interval: float = 1.1, max_retries: int = 4, max_requests: int | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.client, self.model = client, model
        self.min_interval, self.max_retries, self.max_requests = min_interval, max_retries, max_requests
        self._sleep, self._clock = sleep, clock
        self._last_request = float("-inf")
        self.requests = 0                      # solicitudes reales (consumen cuota)
        self.cache_hits = 0
        self.latencies: list[float] = []
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[int]] = self._load_cache()

    # ---- API ----
    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        if not docs:
            return []
        key = self._key(query, docs)
        if key in self._cache:
            self.cache_hits += 1
            order = self._cache[key]
        else:
            order = self._request(query, docs, top_n)
            self._cache[key] = order
            self._append_cache(key, order)
        return [docs[i] for i in order[:top_n]]

    @property
    def mean_latency_ms(self) -> float:
        return 1000 * sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def stats(self) -> dict:
        return {"model": self.model, "requests": self.requests, "cache_hits": self.cache_hits,
                "mean_latency_ms": round(self.mean_latency_ms)}

    # ---- internos ----
    def _key(self, query: str, docs: list[Document]) -> str:
        ids = "\x1f".join(str(d.metadata.get("chunk_id", i)) for i, d in enumerate(docs))
        return hashlib.sha1(f"{self.model}\x1e{query}\x1e{ids}".encode()).hexdigest()

    def _request(self, query: str, docs: list[Document], top_n: int) -> list[int]:
        """Devuelve los índices de `docs` ordenados por relevancia (los `top_n` primeros, o todos)."""
        payload = [{"id": str(d.metadata.get("chunk_id", i)), "text": d.page_content} for i, d in enumerate(docs)]
        for attempt in range(self.max_retries + 1):
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise RerankQuotaError(f"Se alcanzó el tope de {self.max_requests} solicitudes de reranking "
                                       "(--max-rerank-requests). Los resultados ya calculados quedaron en caché.")
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            start = self._clock()
            self._last_request = start
            self.requests += 1
            try:
                result = self.client.inference.rerank(
                    model=self.model, query=query, documents=payload, rank_fields=["text"],
                    return_documents=False, top_n=len(payload), parameters={"truncate": "END"})
            except Exception as exc:                      # noqa: BLE001 - se reclasifica abajo
                if getattr(exc, "status", None) != 429:
                    raise
                if "month" in str(exc).lower() or "limit for the current" in str(exc).lower():
                    raise RerankQuotaError(f"Cuota mensual de reranking agotada: {exc}") from exc
                if attempt == self.max_retries:
                    raise
                self._sleep(2.0 ** attempt)               # 429 por minuto: backoff exponencial
                continue
            self.latencies.append(self._clock() - start)
            return [item.index for item in sorted(result.data, key=lambda r: -r.score)]
        raise AssertionError("inalcanzable")

    def _load_cache(self) -> dict[str, list[int]]:
        if not (self._cache_path and self._cache_path.exists()):
            return {}
        rows = (json.loads(line) for line in self._cache_path.read_text(encoding="utf-8").splitlines() if line)
        return {row["key"]: row["order"] for row in rows}

    def _append_cache(self, key: str, order: list[int]) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "order": order}) + "\n")
