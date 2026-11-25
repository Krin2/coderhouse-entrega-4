import math

import pytest

from src.evaluation.metrics import (
    ndcg_at_k,
    normalized_precision_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_hit_en_top_k():
    assert recall_at_k(["a", "x", "a", "y", "z"], "a", 5) == 1.0


def test_precision_cuenta_chunks_relevantes():
    assert precision_at_k(["a", "x", "a", "y", "z"], "a", 5) == 0.4


def test_sin_aciertos():
    assert recall_at_k(["x", "y"], "a", 5) == 0.0
    assert precision_at_k(["x", "y"], "a", 5) == 0.0
    assert reciprocal_rank(["x", "y"], "a", 5) == 0.0
    assert ndcg_at_k(["x", "y"], "a", 5, n_relevant=2) == 0.0


def test_respeta_k():
    ids = ["x", "y", "z", "a"]                    # el acierto está en la posición 4
    assert recall_at_k(ids, "a", 3) == 0.0
    assert reciprocal_rank(ids, "a", 3) == 0.0
    assert reciprocal_rank(ids, "a", 4) == 0.25


def test_todos_relevantes():
    assert precision_at_k(["a"] * 5, "a", 5) == 1.0


def test_lista_corta_no_infla_precision():
    assert precision_at_k(["a"], "a", 5) == 0.2   # se divide por k, no por len


def test_mrr_usa_el_primer_relevante():
    assert reciprocal_rank(["x", "a", "a"], "a", 5) == 0.5


def test_precision_normalizada_corrige_techo_por_pocos_chunks():
    ids = ["a", "x", "y", "z", "w"]
    assert precision_at_k(ids, "a", 5) == 0.2                       # techo con un solo chunk
    assert normalized_precision_at_k(ids, "a", 5, n_relevant=1) == 1.0
    assert normalized_precision_at_k(ids, "a", 5, n_relevant=10) == 0.2


def test_ndcg_perfecto_y_degradado():
    assert ndcg_at_k(["a", "x"], "a", 5, n_relevant=1) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], "a", 5, n_relevant=1) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["a", "a", "x", "x", "x"], "a", 5, n_relevant=2) == pytest.approx(1.0)


def test_entradas_inconsistentes_no_superan_1():
    # 5 aciertos cuando solo existen 3 chunks relevantes: se ignoran los sobrantes
    assert ndcg_at_k(["a"] * 5, "a", 5, n_relevant=3) == pytest.approx(1.0)
    assert normalized_precision_at_k(["a"] * 5, "a", 5, n_relevant=3) == 1.0


def test_ndcg_en_rango():
    for ids in (["a"] * 5, ["x"] * 5, ["x", "a", "a", "x", "a"]):
        assert 0.0 <= ndcg_at_k(ids, "a", 5, n_relevant=3) <= 1.0


@pytest.mark.parametrize("fn", [recall_at_k, precision_at_k, reciprocal_rank])
def test_k_invalido(fn):
    with pytest.raises(ValueError):
        fn(["a"], "a", 0)


def test_n_relevant_invalido():
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], "a", 5, n_relevant=0)
    with pytest.raises(ValueError):
        normalized_precision_at_k(["a"], "a", 5, n_relevant=0)
