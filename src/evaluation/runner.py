"""Ejecuta el golden set contra distintas estrategias, ajusta pesos en `dev` y mide en `test`."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable

from src.evaluation.metrics import (
    ndcg_at_k,
    normalized_precision_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.evaluation.stats import bootstrap_ci, paired_bootstrap

GRID = [round(0.1 * i, 1) for i in range(1, 10)]          # peso de BM25; el vectorial es 1 - w
METRICS = ("recall", "precision", "norm_precision", "mrr", "ndcg")
VALID_TIPOS = {"exacto", "parafrasis", "ambigua"}
VALID_SPLITS = {"dev", "test"}

BM25, VECTOR, HYBRID = "Solo BM25", "Solo vectorial", "Híbrido 0.5/0.5"
HYBRID_ORACLE = "Híbrido 0.5/0.5 + filtro categoría*"
VECTOR_ORACLE = "Solo vectorial + filtro categoría*"
HYBRID_RERANK = "Híbrido 0.5/0.5 + rerank"
VECTOR_RERANK = "Solo vectorial + rerank"


# ---------- golden set ----------
def load_golden(path: Path) -> list[dict]:
    golden = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_golden(golden)
    return golden


def validate_golden(golden: list[dict]) -> None:
    if not golden:
        raise ValueError("Golden set vacío")
    seen = set()
    for item in golden:
        missing = {"pregunta", "documento_id_esperado"} - item.keys()
        if missing:
            raise ValueError(f"Faltan campos {sorted(missing)} en {item}")
        if item["pregunta"] in seen:
            raise ValueError(f"Pregunta duplicada: {item['pregunta']!r}")
        seen.add(item["pregunta"])
        if "tipo" in item and item["tipo"] not in VALID_TIPOS:
            raise ValueError(f"tipo inválido {item['tipo']!r}; usar {sorted(VALID_TIPOS)}")
        if "split" in item and item["split"] not in VALID_SPLITS:
            raise ValueError(f"split inválido {item['split']!r}; usar {sorted(VALID_SPLITS)}")


def category_of(doc_id: str) -> str:
    """Categoría que asigna la ingesta: primera carpeta, o 'general' para archivos en la raíz."""
    return doc_id.split("/")[0] if "/" in doc_id else "general"


def relevant_counts(corpus) -> Counter:
    """Cuántos chunks tiene cada documento (cota de lo alcanzable en el top-k)."""
    return Counter(d.metadata["doc_id"] for d in corpus)


# ---------- evaluación de un método ----------
def evaluate(fn: Callable, k: int, golden: list[dict], n_relevant: Counter | dict,
             with_category: bool = False) -> list[dict]:
    """`fn(pregunta[, category=...]) -> list[Document]`. Devuelve una fila por pregunta con todas las
    métricas. Con `with_category` se usa la categoría del documento esperado (oráculo): cota superior
    del filtro, no un uso realista."""
    rows = []
    for item in golden:
        expected = item["documento_id_esperado"]
        question = item["pregunta"]
        docs = fn(question, category=category_of(expected)) if with_category else fn(question)
        ids = [d.metadata["doc_id"] for d in docs]
        n_rel = max(1, n_relevant.get(expected, 0))
        rows.append({
            "pregunta": question, "esperado": expected,
            "tipo": item.get("tipo", "sin_tipo"), "split": item.get("split", "test"),
            "recall": recall_at_k(ids, expected, k),
            "precision": precision_at_k(ids, expected, k),
            "norm_precision": normalized_precision_at_k(ids, expected, k, n_rel),
            "mrr": reciprocal_rank(ids, expected, k),
            "ndcg": ndcg_at_k(ids, expected, k, n_rel),
        })
    return rows


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {m: sum(r[m] for r in rows) / len(rows) for m in METRICS} if rows else {}


def subset(rows: list[dict], name: str) -> list[dict]:
    return rows if name == "all" else [r for r in rows if r["split"] == name]


# ---------- ajuste de pesos (solo con el split dev) ----------
def select_weights(rag, golden: list[dict], k: int, n_relevant, grid=GRID):
    """Barre el peso de BM25 sobre `dev` y elige por MRR (desempate: nDCG, luego cercanía a 0.5).
    Devuelve ((bm25, vector), tabla). Sin split dev conserva 0.5/0.5."""
    dev = [g for g in golden if g.get("split") == "dev"]
    if not dev:
        return (0.5, 0.5), []
    table = []
    for w in grid:
        rag.set_weights((w, round(1 - w, 1)))
        m = mean_metrics(evaluate(rag.retrieve, k, dev, n_relevant))
        table.append({"bm25": w, "vector": round(1 - w, 1), **m})
    rag.set_weights((0.5, 0.5))
    best = max(table, key=lambda r: (r["mrr"], r["ndcg"], -abs(r["bm25"] - 0.5)))
    return (best["bm25"], best["vector"]), table


# ---------- comparación completa ----------
def run_comparison(rag, golden: list[dict], k: int, *, n_bootstrap: int = 2000, seed: int = 0) -> dict:
    n_rel = relevant_counts(rag.corpus)
    weights, search = select_weights(rag, golden, k, n_rel)

    methods = {
        BM25: evaluate(rag.retrieve_lexical, k, golden, n_rel),
        VECTOR: evaluate(rag.retrieve_semantic, k, golden, n_rel),
    }
    rag.set_weights((0.5, 0.5))
    methods[HYBRID] = evaluate(rag.retrieve, k, golden, n_rel)
    tuned = HYBRID
    if weights != (0.5, 0.5):
        tuned = f"Híbrido ajustado en dev ({weights[0]:.1f}/{weights[1]:.1f})"
        rag.set_weights(weights)
        methods[tuned] = evaluate(rag.retrieve, k, golden, n_rel)
        rag.set_weights((0.5, 0.5))
    reranker = getattr(rag, "reranker", None)
    if reranker is not None:          # mismos candidatos que HYBRID / VECTOR, reordenados por el reranker
        methods[HYBRID_RERANK] = evaluate(rag.retrieve_reranked, k, golden, n_rel)
        methods[VECTOR_RERANK] = evaluate(rag.retrieve_semantic_reranked, k, golden, n_rel)
    methods[HYBRID_ORACLE] = evaluate(rag.retrieve, k, golden, n_rel, with_category=True)
    methods[VECTOR_ORACLE] = evaluate(rag.retrieve_semantic, k, golden, n_rel, with_category=True)

    results = {
        "k": k, "seed": seed, "n_bootstrap": n_bootstrap,
        "n_questions": {name: len(subset(methods[BM25], name)) for name in ("all", "dev", "test")},
        "weights_search": search, "tuned_weights": list(weights), "tuned_name": tuned,
        "methods": methods,
        "rerank": reranker.stats() if hasattr(reranker, "stats") else None,
    }
    results["analysis"] = analyze(results)
    return results


def analyze(results: dict) -> dict:
    """Intervalos de confianza por método, comparaciones pareadas y desglose por tipo de pregunta."""
    methods, tuned = results["methods"], results["tuned_name"]
    n_boot, seed = results["n_bootstrap"], results["seed"]
    summary: dict = {}
    for part in ("test", "all"):
        summary[part] = {}
        for name, rows in methods.items():
            sub = subset(rows, part)
            if not sub or (part == "all" and name == tuned and tuned != HYBRID):
                continue                               # el ajustado se mide solo en test (no hay fuga)
            entry = {"n": len(sub)}
            for m in METRICS:
                mean, lo, hi = bootstrap_ci([r[m] for r in sub], n_resamples=n_boot, seed=seed)
                entry[m] = {"mean": mean, "lo": lo, "hi": hi}
            summary[part][name] = entry

    pairs = [("test", HYBRID, VECTOR), ("test", HYBRID, BM25), ("all", HYBRID, VECTOR), ("all", HYBRID, BM25)]
    if tuned != HYBRID:
        pairs += [("test", tuned, VECTOR), ("test", tuned, HYBRID)]
    if HYBRID_RERANK in methods:
        for part in ("test", "all"):
            pairs += [(part, HYBRID_RERANK, HYBRID), (part, HYBRID_RERANK, VECTOR),
                      (part, VECTOR_RERANK, VECTOR), (part, HYBRID_RERANK, VECTOR_RERANK)]
    comparisons = []
    for part, a, b in pairs:
        ra, rb = subset(methods[a], part), subset(methods[b], part)
        if not ra:
            continue
        for metric in ("recall", "mrr", "ndcg"):
            comparisons.append({
                "subset": part, "a": a, "b": b, "metric": metric,
                **paired_bootstrap([r[metric] for r in ra], [r[metric] for r in rb], n_resamples=n_boot, seed=seed),
            })

    by_type: dict = {}
    for tipo in sorted({r["tipo"] for r in methods[HYBRID]}):
        by_type[tipo] = {}
        for name in (BM25, VECTOR, HYBRID, HYBRID_RERANK, VECTOR_RERANK):
            if name not in methods:
                continue
            rows = [r for r in methods[name] if r["tipo"] == tipo]
            by_type[tipo][name] = {"n": len(rows), **mean_metrics(rows)}
    return {"summary": summary, "comparisons": comparisons, "by_type": by_type}
