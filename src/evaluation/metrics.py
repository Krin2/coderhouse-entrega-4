"""Métricas de recuperación como funciones puras. Relevante = chunk cuyo doc_id es el esperado."""
from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], expected: str, k: int) -> float:
    """1.0 si algún chunk relevante aparece en el top-k, 0.0 si no."""
    return 1.0 if expected in retrieved_doc_ids[:k] else 0.0


def precision_at_k(retrieved_doc_ids: list[str], expected: str, k: int) -> float:
    """Fracción de los k recuperados que son relevantes."""
    return sum(1 for i in retrieved_doc_ids[:k] if i == expected) / k
