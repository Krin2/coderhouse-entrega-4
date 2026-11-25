import dataclasses

import pytest

from src.cli import build_parser
from src.settings import Settings


def test_import_no_lee_entorno(monkeypatch):
    monkeypatch.setenv("INDEX_NAME", "otro")
    assert Settings().index_name == "rag-fastapi-docs"        # el default no depende del entorno


def test_from_env_lee_variables(monkeypatch):
    monkeypatch.setenv("INDEX_NAME", "mi-indice")
    monkeypatch.setenv("NAMESPACE", "mi-ns")
    s = Settings.from_env()
    assert (s.index_name, s.namespace) == ("mi-indice", "mi-ns")


def test_from_env_overrides():
    assert Settings.from_env(top_k=9).top_k == 9


def test_claves_ocultas_en_repr():
    assert "sk-secreto" not in repr(Settings(openai_api_key="sk-secreto", pinecone_api_key="pc-secreto"))


def test_inmutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().top_k = 1


def test_require_keys(monkeypatch):
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY, OPENAI_API_KEY"):
        Settings().require_keys()
    Settings(pinecone_api_key="a", openai_api_key="b").require_keys()


def test_cache_por_namespace():
    assert Settings(namespace="a").corpus_cache != Settings(namespace="b").corpus_cache
    assert Settings(namespace="a").corpus_cache.name == "a.jsonl"


def test_cli_parsea_comandos():
    p = build_parser()
    assert p.parse_args(["query", "hola", "--category", "advanced", "-k", "3"]).category == "advanced"
    assert p.parse_args(["evaluate"]).k == 5
    with pytest.raises(SystemExit):
        p.parse_args([])
    ing = p.parse_args(["ingest", "--with-code", "--namespace", "x"])
    assert ing.with_code and ing.namespace == "x" and not p.parse_args(["ingest"]).with_code
    cmp = p.parse_args(["compare-reports", "a.json", "b.json", "--out", "o.md"])
    assert (cmp.base, cmp.other, cmp.out) == ("a.json", "b.json", "o.md")
