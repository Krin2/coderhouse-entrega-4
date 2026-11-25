import pytest

from src.evaluation.stats import bootstrap_ci, paired_bootstrap


def test_ci_contiene_la_media_y_es_reproducible():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    mean, lo, hi = bootstrap_ci(values, seed=1)
    assert lo <= mean <= hi and mean == pytest.approx(0.625)
    assert bootstrap_ci(values, seed=1) == (mean, lo, hi)


def test_ci_constante_es_degenerado():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_ci_mas_ancho_con_menos_datos():
    wide = bootstrap_ci([1, 0, 1, 0], seed=0)
    narrow = bootstrap_ci([1, 0, 1, 0] * 25, seed=0)
    assert (wide[2] - wide[1]) > (narrow[2] - narrow[1])


def test_ci_vacio():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_pareado_detecta_mejora_clara():
    a, b = [1.0] * 30, [0.0] * 30
    res = paired_bootstrap(a, b, seed=0)
    assert res["diff"] == 1.0 and res["p"] == 0.0 and res["lo"] == 1.0


def test_pareado_sin_diferencia():
    x = [1.0, 0.0, 1.0, 0.0]
    res = paired_bootstrap(x, list(x), seed=0)
    assert res["diff"] == 0.0 and res["p"] == 1.0                 # a no supera a b


def test_pareado_ruido_no_es_significativo():
    a = [1, 0, 1, 0, 1, 0, 1, 0]
    b = [0, 1, 1, 0, 0, 1, 1, 0]
    res = paired_bootstrap(a, b, seed=0)
    assert res["lo"] <= 0 <= res["hi"] and res["p"] > 0.05


def test_pareado_largos_distintos():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])
