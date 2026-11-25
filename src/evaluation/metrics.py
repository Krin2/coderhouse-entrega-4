"""Métricas de recuperación como funciones puras.

Relevante = chunk cuyo `doc_id` es el documento esperado. `n_relevant` es cuántos chunks relevantes
existen en el corpus: acota lo que es alcanzable en el top-k (un documento de 1 chunk nunca puede dar
Precision@5 > 0.2).
"""
from __future__ import annotations

from math import log2


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k debe ser >= 1")


def recall_at_k(retrieved_doc_ids: list[str], expected: str, k: int) -> float:
    """1.0 si algún chunk relevante aparece en el top-k, 0.0 si no."""
    _check_k(k)
    return 1.0 if expected in retrieved_doc_ids[:k] else 0.0


def precision_at_k(retrieved_doc_ids: list[str], expected: str, k: int) -> float:
    """Fracción de los k recuperados que son relevantes."""
    _check_k(k)
    return sum(1 for i in retrieved_doc_ids[:k] if i == expected) / k


def normalized_precision_at_k(retrieved_doc_ids: list[str], expected: str, k: int, n_relevant: int) -> float:
    """Precision@k dividida por su máximo alcanzable, min(k, n_relevant)/k. Vale 1.0 si el top-k
    contiene todos los chunks relevantes que podía contener."""
    _check_k(k)
    if n_relevant < 1:
        raise ValueError("n_relevant debe ser >= 1")
    hits = sum(1 for i in retrieved_doc_ids[:k] if i == expected)
    ceiling = min(k, n_relevant)
    return min(hits, ceiling) / ceiling


def reciprocal_rank(retrieved_doc_ids: list[str], expected: str, k: int) -> float:
    """1/posición del primer relevante dentro del top-k (0.0 si no aparece). Promediada da el MRR."""
    _check_k(k)
    for position, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id == expected:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], expected: str, k: int, n_relevant: int) -> float:
    """nDCG@k con relevancia binaria; el ideal coloca min(k, n_relevant) relevantes al comienzo."""
    _check_k(k)
    if n_relevant < 1:
        raise ValueError("n_relevant debe ser >= 1")
    dcg, found = 0.0, 0
    for pos, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id == expected and found < n_relevant:      # nunca cuenta más relevantes de los que existen
            dcg += 1 / log2(pos + 1)
            found += 1
    ideal = sum(1 / log2(pos + 1) for pos in range(1, min(k, n_relevant) + 1))
    return dcg / ideal
