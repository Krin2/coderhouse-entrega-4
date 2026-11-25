import pytest
from langchain_core.documents import Document

from src.evaluation import runner
from src.evaluation.report import to_markdown, write_reports
from src.evaluation.runner import (
    category_of,
    evaluate,
    load_golden,
    relevant_counts,
    run_comparison,
    select_weights,
    validate_golden,
)
from src.settings import Settings


def doc(doc_id):
    return Document(page_content=doc_id, metadata={"doc_id": doc_id, "category": category_of(doc_id)})


class FakeRag:
    """Retriever falso cuyo acierto depende del peso de BM25 (óptimo en 0.3)."""

    def __init__(self, corpus):
        self.corpus, self.weights = corpus, (0.5, 0.5)

    def set_weights(self, w): self.weights = tuple(w)

    def _answer(self, q, good_first):
        right = doc(self.expected[q])
        return [right, doc("x/otro")] if good_first else [doc("x/otro"), right]

    def retrieve(self, q, category=None): return self._answer(q, abs(self.weights[0] - 0.3) < 1e-9)
    def retrieve_semantic(self, q, category=None): return self._answer(q, True)
    def retrieve_lexical(self, q, category=None): return [doc("x/otro")]


GOLDEN = [
    {"pregunta": f"q{i}", "documento_id_esperado": f"tutorial/d{i}", "tipo": t, "split": s}
    for i, (t, s) in enumerate([("exacto", "dev"), ("exacto", "test"), ("parafrasis", "dev"),
                                ("parafrasis", "test"), ("ambigua", "dev"), ("ambigua", "test")])
]


@pytest.fixture
def rag():
    corpus = [doc(f"tutorial/d{i}") for i in range(6)] + [doc("x/otro")]
    fake = FakeRag(corpus)
    fake.expected = {g["pregunta"]: g["documento_id_esperado"] for g in GOLDEN}
    return fake


def test_category_of():
    assert category_of("advanced/events") == "advanced"
    assert category_of("async") == "general"             # archivos en la raíz => 'general'


def test_relevant_counts():
    assert relevant_counts([doc("a/x"), doc("a/x"), doc("b/y")]) == {"a/x": 2, "b/y": 1}


def test_validate_golden_rechaza_errores():
    ok = {"pregunta": "q", "documento_id_esperado": "a/b"}
    validate_golden([ok])
    with pytest.raises(ValueError, match="vacío"):
        validate_golden([])
    with pytest.raises(ValueError, match="Faltan"):
        validate_golden([{"pregunta": "q"}])
    with pytest.raises(ValueError, match="duplicada"):
        validate_golden([ok, dict(ok)])
    with pytest.raises(ValueError, match="tipo"):
        validate_golden([{**ok, "tipo": "raro"}])
    with pytest.raises(ValueError, match="split"):
        validate_golden([{**ok, "split": "val"}])


def test_golden_set_real_es_valido_y_balanceado():
    golden = load_golden(Settings().golden_set)
    assert len(golden) >= 40
    for tipo in runner.VALID_TIPOS:
        for split in runner.VALID_SPLITS:
            assert sum(1 for g in golden if g["tipo"] == tipo and g["split"] == split) >= 3, (tipo, split)


def test_evaluate_calcula_todas_las_metricas():
    golden = [{"pregunta": "q", "documento_id_esperado": "a/b"}]
    row = evaluate(lambda q: [doc("x/y"), doc("a/b")], 5, golden, {"a/b": 1})[0]
    assert row["recall"] == 1.0 and row["mrr"] == 0.5 and row["precision"] == 0.2
    assert row["norm_precision"] == 1.0 and 0 < row["ndcg"] < 1
    assert row["split"] == "test" and row["tipo"] == "sin_tipo"      # defaults sin metadata


def test_evaluate_con_categoria_la_deriva_del_doc_esperado():
    seen = []
    golden = [{"pregunta": "q", "documento_id_esperado": "advanced/events"},
              {"pregunta": "q2", "documento_id_esperado": "async"}]
    evaluate(lambda q, category: seen.append(category) or [], 5, golden, {}, with_category=True)
    assert seen == ["advanced", "general"]


def test_select_weights_usa_solo_dev_y_elige_el_mejor(rag):
    calls = []
    original = rag.retrieve

    def spy(q, category=None):
        calls.append(q)
        return original(q, category)

    rag.retrieve = spy
    weights, table = select_weights(rag, GOLDEN, 5, relevant_counts(rag.corpus))
    assert weights == (0.3, 0.7) and len(table) == len(runner.GRID)
    assert set(calls) == {"q0", "q2", "q4"}                       # ninguna pregunta de test
    assert rag.weights == (0.5, 0.5)                              # restaura los pesos


def test_select_weights_sin_dev_conserva_default(rag):
    test_only = [{**g, "split": "test"} for g in GOLDEN]
    assert select_weights(rag, test_only, 5, {}) == ((0.5, 0.5), [])


def test_run_comparison_estructura_y_sin_fuga(rag):
    res = run_comparison(rag, GOLDEN, 5, n_bootstrap=200, seed=0)
    assert res["n_questions"] == {"all": 6, "dev": 3, "test": 3}
    assert res["tuned_weights"] == [0.3, 0.7] and res["tuned_name"].startswith("Híbrido ajustado")
    tuned = res["tuned_name"]
    assert tuned in res["analysis"]["summary"]["test"]
    assert tuned not in res["analysis"]["summary"]["all"]          # el ajustado no se mide donde se ajustó
    assert res["analysis"]["summary"]["test"][tuned]["mrr"]["mean"] == 1.0
    assert res["analysis"]["summary"]["test"][runner.HYBRID]["mrr"]["mean"] == 0.5
    assert {c["metric"] for c in res["analysis"]["comparisons"]} == {"recall", "mrr"}
    assert set(res["analysis"]["by_type"]) == {"exacto", "parafrasis", "ambigua"}


def test_reporte_markdown_y_archivos(rag, tmp_path):
    res = run_comparison(rag, GOLDEN, 5, n_bootstrap=100, seed=0)
    md = to_markdown(res)
    for fragment in ("# Reporte de evaluación", "Ajuste de pesos", "Comparaciones pareadas", "Por tipo", "MRR"):
        assert fragment in md
    md_path, json_path = write_reports(res, tmp_path / "reports")
    assert md_path.read_text(encoding="utf-8") == md and json_path.exists()
