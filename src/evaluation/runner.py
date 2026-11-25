"""Ejecuta el golden set contra distintas estrategias y arma el reporte comparativo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.evaluation.metrics import precision_at_k, recall_at_k

WEIGHTS = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]      # (bm25, vector)
DEFAULT_LABEL = "Híbrido bm25=0.5 vec=0.5"


def load_golden(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def evaluate(fn: Callable, k: int, golden: list[dict], with_category: bool = False) -> dict:
    """`fn(pregunta[, category=...]) -> list[Document]`. Con `with_category` se usa la categoría del
    documento esperado (oráculo): cota superior del filtro, no un uso realista."""
    rows = []
    for item in golden:
        expected = item["documento_id_esperado"]
        docs = fn(item["pregunta"], category=expected.split("/")[0]) if with_category else fn(item["pregunta"])
        ids = [d.metadata["doc_id"] for d in docs]
        rows.append((item["pregunta"], expected, recall_at_k(ids, expected, k), precision_at_k(ids, expected, k)))
    n = len(rows)
    return {"rows": rows, "recall": sum(r[2] for r in rows) / n, "precision": sum(r[3] for r in rows) / n}


def compare(rag, golden: list[dict], k: int) -> dict[str, dict]:
    """Solo BM25, solo vectorial, híbrido con varios pesos y las variantes con filtro de categoría."""
    results = {
        "Solo BM25": evaluate(rag.retrieve_lexical, k, golden),
        "Solo vectorial": evaluate(rag.retrieve_semantic, k, golden),
    }
    for bm25_w, vec_w in WEIGHTS:
        rag.set_weights((bm25_w, vec_w))
        results[f"Híbrido bm25={bm25_w} vec={vec_w}"] = evaluate(rag.retrieve, k, golden)
    rag.set_weights((0.5, 0.5))
    results["Híbrido 0.5/0.5 + filtro categoría*"] = evaluate(rag.retrieve, k, golden, with_category=True)
    results["Solo vectorial + filtro categoría*"] = evaluate(rag.retrieve_semantic, k, golden, with_category=True)
    return results


def format_report(results: dict[str, dict], k: int, n: int) -> str:
    lines = [f"\n=== Detalle: {DEFAULT_LABEL} ({n} preguntas) ==="]
    for q, exp, r, p in results[DEFAULT_LABEL]["rows"]:
        lines.append(f"  R@{k}={r:.0f}  P@{k}={p:.2f}  {exp:<42} {q[:50]}")
    lines += [f"\n=== Resumen (n={n}) ===", f"{'Método':<38}{'Recall@' + str(k):>10}{'Precision@' + str(k):>14}"]
    lines += [f"{name:<38}{res['recall']:>10.2f}{res['precision']:>14.2f}" for name, res in results.items()]
    lines.append("* categoría conocida de antemano (oráculo): cota superior del filtro, no uso real.")
    return "\n".join(lines)
