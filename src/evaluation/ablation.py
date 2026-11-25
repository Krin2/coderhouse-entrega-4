"""Compara dos corridas de evaluación (p. ej. índice sin código vs con código) sobre las mismas preguntas."""
from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.runner import subset
from src.evaluation.stats import paired_bootstrap

METRICS = ("recall", "mrr", "ndcg")


def load_results(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aligned(base_rows: list[dict], other_rows: list[dict], part: str):
    by_question = {r["pregunta"]: r for r in subset(other_rows, part)}
    base = subset(base_rows, part)
    if {r["pregunta"] for r in base} != by_question.keys():
        raise ValueError("Las dos corridas no usaron las mismas preguntas: no son comparables")
    return base, [by_question[r["pregunta"]] for r in base]


def compare_runs(base: dict, other: dict, *, n_bootstrap: int = 2000, seed: int = 0) -> list[dict]:
    """Por cada método presente en ambas corridas: diferencia pareada other − base (IC 95 % y p)."""
    skip_in_all = {base.get("tuned_name"), other.get("tuned_name")} - {"Híbrido 0.5/0.5"}
    out = []
    for name in base["methods"].keys() & other["methods"].keys():
        for part in ("test", "all"):
            if part == "all" and name in skip_in_all:
                continue                                  # el ajustado en dev no se mide donde se ajustó
            rows_b, rows_o = _aligned(base["methods"][name], other["methods"][name], part)
            if not rows_b:
                continue
            for metric in METRICS:
                xs, ys = [r[metric] for r in rows_o], [r[metric] for r in rows_b]
                out.append({
                    "method": name, "subset": part, "metric": metric, "n": len(rows_b),
                    "base": sum(ys) / len(ys), "other": sum(xs) / len(xs),
                    **paired_bootstrap(xs, ys, n_resamples=n_bootstrap, seed=seed),
                })
    order = {"Solo BM25": 0, "Solo vectorial": 1, "Híbrido 0.5/0.5": 2}
    return sorted(out, key=lambda r: (
        r["subset"] != "all", order.get(r["method"], 9), r["method"], METRICS.index(r["metric"])))


def to_markdown(rows: list[dict], base_label: str, other_label: str) -> str:
    lines = [f"# Ablación: {other_label} vs {base_label}", "",
             f"Δ = {other_label} − {base_label} sobre las mismas preguntas (bootstrap pareado, IC 95 %). "
             "`p` = fracción de remuestreos en que no hay mejora (p bajo ⇒ la mejora es creíble).", "",
             f"| Subconjunto | Método | Métrica | {base_label} | {other_label} | Δ | IC 95 % | p |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['subset']} (n={r['n']}) | {r['method']} | {r['metric']} | {r['base']:.3f} | "
                     f"{r['other']:.3f} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.3f} |")
    return "\n".join(lines) + "\n"
