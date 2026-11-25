import pytest

from src.evaluation.metrics import precision_at_k, recall_at_k
from src.evaluation.runner import evaluate, load_golden


def test_recall_hit_en_top_k():
    assert recall_at_k(["a", "x", "a", "y", "z"], "a", 5) == 1.0


def test_precision_cuenta_chunks_relevantes():
    assert precision_at_k(["a", "x", "a", "y", "z"], "a", 5) == 0.4


def test_sin_aciertos():
    assert recall_at_k(["x", "y"], "a", 5) == 0.0
    assert precision_at_k(["x", "y"], "a", 5) == 0.0


def test_respeta_k():
    assert recall_at_k(["x", "y", "z", "a"], "a", 3) == 0.0       # el acierto está en la posición 4


def test_todos_relevantes():
    assert precision_at_k(["a"] * 5, "a", 5) == 1.0


def test_lista_corta_no_infla_precision():
    assert precision_at_k(["a"], "a", 5) == 0.2                    # se divide por k, no por len


class _Doc:
    def __init__(self, doc_id): self.metadata = {"doc_id": doc_id}


def test_evaluate_promedia():
    golden = [{"pregunta": "q1", "documento_id_esperado": "a"}, {"pregunta": "q2", "documento_id_esperado": "b"}]
    fn = lambda q: [_Doc("a")] * 2 + [_Doc("z")] * 3 if q == "q1" else [_Doc("z")] * 5
    res = evaluate(fn, 5, golden)
    assert res["recall"] == 0.5 and res["precision"] == pytest.approx(0.2)


def test_evaluate_con_categoria_la_deriva_del_doc_esperado():
    seen = []
    golden = [{"pregunta": "q", "documento_id_esperado": "advanced/events"}]
    evaluate(lambda q, category: seen.append(category) or [], 5, golden, with_category=True)
    assert seen == ["advanced"]


def test_golden_set_valido():
    from src.settings import Settings
    golden = load_golden(Settings().golden_set)
    assert len(golden) >= 5
    assert all(set(g) == {"pregunta", "documento_id_esperado"} for g in golden)
    assert len({g["pregunta"] for g in golden}) == len(golden)
