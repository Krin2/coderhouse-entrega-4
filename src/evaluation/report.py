"""Formato del reporte de evaluación (Markdown para consola y archivo, JSON para máquinas)."""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.runner import BM25, HYBRID, VECTOR


def _ci(entry: dict) -> str:
    return f"{entry['mean']:.2f} [{entry['lo']:.2f}, {entry['hi']:.2f}]"


def _summary_table(summary: dict, k: int) -> list[str]:
    lines = [f"| Método | Recall@{k} | MRR | nDCG@{k} | Precision@{k} | Precision@{k} normalizada |",
             "|---|---|---|---|---|---|"]
    for name, e in summary.items():
        lines.append(f"| {name} (n={e['n']}) | {_ci(e['recall'])} | {_ci(e['mrr'])} | {_ci(e['ndcg'])} "
                     f"| {_ci(e['precision'])} | {_ci(e['norm_precision'])} |")
    return lines


def to_markdown(results: dict) -> str:
    k, n, a = results["k"], results["n_questions"], results["analysis"]
    lines = [
        "# Reporte de evaluación",
        f"\nk={k} · preguntas: {n['all']} (dev={n['dev']}, test={n['test']}) · "
        f"bootstrap={results['n_bootstrap']} remuestreos, semilla={results['seed']} · IC 95 % entre corchetes.",
    ]

    if results["weights_search"]:
        lines += [f"\n## Ajuste de pesos (solo `dev`, n={n['dev']})", "",
                  "| Peso BM25 | Peso vector | MRR | Recall | nDCG |", "|---|---|---|---|---|"]
        best = tuple(results["tuned_weights"])
        for row in results["weights_search"]:
            mark = " ←" if (row["bm25"], row["vector"]) == best else ""
            lines.append(f"| {row['bm25']:.1f} | {row['vector']:.1f} | {row['mrr']:.3f} | "
                         f"{row['recall']:.2f} | {row['ndcg']:.3f} |{mark}")

    lines += [f"\n## Resultados en `test` (n={n['test']}, no usado para ajustar nada)", ""]
    lines += _summary_table(a["summary"]["test"], k)
    lines += [f"\n## Todo el set (n={n['all']}) — métodos sin ajuste", ""]
    lines += _summary_table(a["summary"]["all"], k)
    lines.append("\n\\* categoría conocida de antemano (oráculo): cota superior del filtro, no uso real.")

    lines += ["\n## Comparaciones pareadas (mismas preguntas)", "",
              "Δ = a − b. `p` = fracción de remuestreos en que a no supera a b (p bajo ⇒ evidencia de que a es mejor).",
              "", "| Subconjunto | a | b | Métrica | Δ | IC 95 % | p |", "|---|---|---|---|---|---|---|"]
    for c in a["comparisons"]:
        lines.append(f"| {c['subset']} | {c['a']} | {c['b']} | {c['metric']} | {c['diff']:+.3f} | "
                     f"[{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p']:.3f} |")

    lines += ["\n## Por tipo de pregunta (todo el set)", "",
              f"| Tipo | n | {BM25} (R / MRR) | {VECTOR} (R / MRR) | {HYBRID} (R / MRR) |", "|---|---|---|---|---|"]
    for tipo, per in a["by_type"].items():
        cells = " | ".join(f"{per[m]['recall']:.2f} / {per[m]['mrr']:.2f}" for m in (BM25, VECTOR, HYBRID))
        lines.append(f"| {tipo} | {per[HYBRID]['n']} | {cells} |")

    misses = [r for r in results["methods"][HYBRID] if r["recall"] == 0]
    lines += [f"\n## Preguntas que el híbrido 0.5/0.5 no resuelve en el top-{k} ({len(misses)})", ""]
    lines += [f"- `{r['esperado']}` ({r['tipo']}, {r['split']}): {r['pregunta']}" for r in misses] or ["- ninguna"]
    return "\n".join(lines) + "\n"


def write_reports(results: dict, directory: Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    md_path, json_path = directory / "latest.md", directory / "latest.json"
    md_path.write_text(to_markdown(results), encoding="utf-8")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path
