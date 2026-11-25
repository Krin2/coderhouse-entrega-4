import json

import pytest

from src.evaluation.ablation import compare_runs, to_markdown


def _row(q, split, recall, mrr, ndcg=0.5):
    return {"pregunta": q, "split": split, "tipo": "exacto", "esperado": "a/b",
            "recall": recall, "mrr": mrr, "ndcg": ndcg, "precision": 0.2, "norm_precision": 0.2}


def _run(mrrs, tuned="Híbrido 0.5/0.5"):
    rows = [_row(f"q{i}", "test" if i % 2 else "dev", 1.0, m) for i, m in enumerate(mrrs)]
    return {"tuned_name": tuned, "methods": {"Solo BM25": rows, "Híbrido 0.5/0.5": rows}}


def test_mejora_clara_es_significativa():
    out = compare_runs(_run([0.2] * 12), _run([0.9] * 12), n_bootstrap=200)
    mrr = next(r for r in out if r["method"] == "Solo BM25" and r["subset"] == "all" and r["metric"] == "mrr")
    assert mrr["diff"] == pytest.approx(0.7) and mrr["p"] == 0.0 and mrr["n"] == 12


def test_sin_cambios_no_es_mejora():
    out = compare_runs(_run([0.5] * 12), _run([0.5] * 12), n_bootstrap=100)
    assert all(r["diff"] == 0 and r["p"] == 1.0 for r in out)


def test_preguntas_distintas_no_son_comparables():
    other = _run([0.5] * 12)
    other["methods"]["Solo BM25"][0]["pregunta"] = "otra"
    with pytest.raises(ValueError, match="mismas preguntas"):
        compare_runs(_run([0.5] * 12), other, n_bootstrap=50)


def test_ajustado_no_se_compara_en_all():
    tuned = "Híbrido ajustado en dev (0.3/0.7)"
    base, other = _run([0.5] * 12, tuned), _run([0.6] * 12, tuned)
    for run in (base, other):
        run["methods"][tuned] = run["methods"]["Solo BM25"]
    out = compare_runs(base, other, n_bootstrap=50)
    assert {r["subset"] for r in out if r["method"] == tuned} == {"test"}


def test_markdown():
    md = to_markdown(compare_runs(_run([0.2] * 12), _run([0.9] * 12), n_bootstrap=50), "sin código", "con código")
    assert "con código vs sin código" in md and "| all (n=12)" in md and json.dumps(0)  # tabla presente
