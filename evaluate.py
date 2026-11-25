"""Evalúa el recuperador con Precision@k y Recall@k sobre el golden set.

Compara: solo BM25, solo vectorial y el híbrido con distintos pesos (BM25, vector).
"""
import json
import sys
from pathlib import Path

from src import config
from src.rag_system import RAGSystem

GOLDEN = Path("eval/golden_set.json")
WEIGHTS = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]


def score(retrieved_ids: list[str], expected: str, k: int) -> tuple[float, float]:
    """Relevante = chunk cuyo doc_id es el esperado.
    Recall@k: 1 si algún relevante aparece en el top-k. Precision@k: relevantes / k."""
    top = retrieved_ids[:k]
    hits = sum(1 for i in top if i == expected)
    return (1.0 if hits else 0.0), hits / k


def evaluate(fn, k: int, golden: list[dict], with_category: bool = False) -> dict:
    rows = []
    for item in golden:
        if with_category:   # 'oráculo': la categoría sale del documento esperado (cota superior)
            docs = fn(item["pregunta"], category=item["documento_id_esperado"].split("/")[0])
        else:
            docs = fn(item["pregunta"])
        r, p = score([d.metadata["doc_id"] for d in docs], item["documento_id_esperado"], k)
        rows.append((item["pregunta"], item["documento_id_esperado"], r, p))
    n = len(rows)
    return {"rows": rows, "recall": sum(x[2] for x in rows) / n, "precision": sum(x[3] for x in rows) / n}


def main(k: int = config.TOP_K) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8-sig"))
    rag = RAGSystem(k=k)

    results = {
        "Solo BM25": evaluate(lambda q: rag.bm25.invoke(q)[:k], k, golden),
        "Solo vectorial": evaluate(rag.retrieve_semantic, k, golden),
    }
    for w in WEIGHTS:
        rag.ensemble.weights = list(w)
        results[f"Híbrido bm25={w[0]} vec={w[1]}"] = evaluate(rag.retrieve, k, golden)

    rag.ensemble.weights = [0.5, 0.5]
    results["Híbrido 0.5/0.5 + filtro categoría*"] = evaluate(rag.retrieve, k, golden, with_category=True)
    results["Solo vectorial + filtro categoría*"] = evaluate(rag.retrieve_semantic, k, golden, with_category=True)

    # Detalle por pregunta del híbrido 0.5/0.5 (configuración por defecto)
    default = results["Híbrido bm25=0.5 vec=0.5"]
    print(f"\n=== Detalle: híbrido 0.5/0.5 ({len(golden)} preguntas) ===")
    for q, exp, r, p in default["rows"]:
        print(f"  R@{k}={r:.0f}  P@{k}={p:.2f}  {exp:<42} {q[:50]}")

    print(f"\n=== Resumen (n={len(golden)}) ===")
    print(f"{'Método':<38}{'Recall@'+str(k):>10}{'Precision@'+str(k):>14}")
    for name, res in results.items():
        print(f"{name:<38}{res['recall']:>10.2f}{res['precision']:>14.2f}")
    print("* categoría conocida de antemano (oráculo): cota superior del filtro, no uso real.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
