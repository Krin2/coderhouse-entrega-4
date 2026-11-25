from src.ingestion.chunking import extract_section, split_documents
from src.ingestion.loaders import MarkdownLoader


def test_extract_section_quita_ancla():
    assert extract_section("intro\n## Titulo { #titulo }\ntexto") == "Titulo"


def test_extract_section_fallback():
    assert extract_section("solo texto", fallback="previa") == "previa"


def test_loader_metadata(tmp_path):
    (tmp_path / "tutorial").mkdir()
    (tmp_path / "tutorial" / "a.md").write_text("# A\n{* ref.py *}\ntexto", encoding="utf-8")
    (tmp_path / "top.md").write_text("# Top", encoding="utf-8")
    docs = {d.metadata["doc_id"]: d for d in MarkdownLoader(tmp_path).load()}
    assert docs["tutorial/a"].metadata["category"] == "tutorial"
    assert docs["tutorial/a"].metadata["source"] == "tutorial/a.md"
    assert docs["top"].metadata["category"] == "general"
    assert "{*" not in docs["tutorial/a"].page_content


def test_loader_directorio_vacio(tmp_path):
    assert MarkdownLoader(tmp_path).load() == []


def test_chunking_ids_indices_y_secciones(tmp_path):
    body = "# Titulo { #titulo }\n\n" + "palabra " * 700 + "\n\n## Segunda { #segunda }\n\n" + "otra " * 700
    (tmp_path / "doc.md").write_text(body, encoding="utf-8")
    chunks = split_documents(MarkdownLoader(tmp_path).load(), chunk_size=600, chunk_overlap=80)
    assert len(chunks) > 1
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata["chunk_id"] == "doc#0"
    assert chunks[0].metadata["section"] == "Titulo"
    assert chunks[-1].metadata["section"] == "Segunda"
    assert all(c.metadata["category"] == "general" for c in chunks)


def test_chunking_reinicia_contadores_por_documento(tmp_path):
    for name in ("a", "b"):
        (tmp_path / f"{name}.md").write_text(f"# {name}\n\ntexto", encoding="utf-8")
    chunks = split_documents(MarkdownLoader(tmp_path).load(), chunk_size=600, chunk_overlap=80)
    assert [c.metadata["chunk_id"] for c in chunks] == ["a#0", "b#0"]
    assert [c.metadata["section"] for c in chunks] == ["a", "b"]      # la sección no se filtra entre docs
