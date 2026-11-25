"""Evalúa el recuperador (solo vectorial vs híbrido) con Precision@k y Recall@k."""
import json
from pathlib import Path

from src import config
from src.rag_system import RAGSystem

GOLDEN = Path("eval/golden_set.json")


def score(retrieved_ids: list[str], expected: str, k: int) -> tuple[float, float]:
    """Relevante = chunk cuyo doc_id es el esperado.
    Recall@k: 1 si algún relevante aparece en el top-k. Precision@k: relevantes / k."""
    top = retrieved_ids[:k]
    hits = sum(1 for i in top if i == expected)
    return (1.0 if hits else 0.0), hits / k


def evaluate(rag: RAGSystem, fn, k: int, golden: list[dict]) -> dict:
    rows = []
    for item in golden:
        docs = fn(item["pregunta"])
        r, p = score([d.metadata["doc_id"] for d in docs], item["documento_id_esperado"], k)
        rows.append((item["pregunta"], item["documento_id_esperado"], r, p))
    n = len(rows)
    return {"rows": rows, "recall": sum(x[2] for x in rows) / n, "precision": sum(x[3] for x in rows) / n}


def main(k: int = config.TOP_K) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8-sig"))
    rag = RAGSystem(k=k)
    results = {
        "Solo vectorial": evaluate(rag, rag.retrieve_semantic, k, golden),
        "Híbrido (BM25+vector)": evaluate(rag, rag.retrieve, k, golden),
    }
    for name, res in results.items():
        print(f"\n=== {name} ===")
        for q, exp, r, p in res["rows"]:
            print(f"  R@{k}={r:.0f}  P@{k}={p:.2f}  {exp:<32} {q[:55]}")
        print(f"  -> Recall@{k} = {res['recall']:.2f} | Precision@{k} = {res['precision']:.2f}")
    print("\n=== Resumen ===")
    print(f"{'Método':<24}{'Recall@'+str(k):>10}{'Precision@'+str(k):>14}")
    for name, res in results.items():
        print(f"{name:<24}{res['recall']:>10.2f}{res['precision']:>14.2f}")


if __name__ == "__main__":
    main()

