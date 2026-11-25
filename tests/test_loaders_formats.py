import json

import pytest

from src.ingestion.chunking import split_documents
from src.ingestion.loaders import CompositeLoader, JsonLoader, MarkdownLoader, PdfLoader, default_loader
from tests.conftest import make_pdf


# ---------- PDF ----------
def test_pdf_una_pagina_por_documento_con_numero_real(tmp_path):
    (tmp_path / "guias").mkdir()
    make_pdf(tmp_path / "guias" / "manual.pdf", ["Capitulo uno\nsobre cache", "Capitulo dos", "Capitulo tres"])
    docs = PdfLoader(tmp_path).load()
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
    first = docs[0]
    assert "Capitulo uno" in first.page_content and "sobre cache" in first.page_content
    assert first.metadata | {"page": 1} == {"source": "guias/manual.pdf", "doc_id": "guias/manual",
                                            "category": "guias", "page": 1, "total_pages": 3}


def test_pdf_omite_paginas_sin_texto_y_las_registra(tmp_path):
    make_pdf(tmp_path / "scan.pdf", ["hola", "", "adios"])
    loader = PdfLoader(tmp_path)
    assert [d.metadata["page"] for d in loader.load()] == [1, 3]
    assert loader.skipped_pages == [("scan.pdf", 2)]


def test_pdf_categoria_general_en_la_raiz(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["x"])
    assert PdfLoader(tmp_path).load()[0].metadata["category"] == "general"


def test_pdf_corrupto_falla_de_forma_explicita(tmp_path):
    (tmp_path / "roto.pdf").write_bytes(b"esto no es un pdf")
    with pytest.raises(Exception):
        PdfLoader(tmp_path).load()


def test_pdf_la_pagina_llega_a_los_chunks(tmp_path):
    make_pdf(tmp_path / "doc.pdf", ["primera pagina", "segunda pagina"])
    chunks = split_documents(PdfLoader(tmp_path).load(), chunk_size=600, chunk_overlap=80)
    assert [(c.metadata["chunk_id"], c.metadata["page"]) for c in chunks] == [("doc#0", 1), ("doc#1", 2)]
    assert all("section" not in c.metadata for c in chunks)               # sin encabezados: no se guarda vacío


# ---------- JSON ----------
def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_json_registros_con_metadata_y_tags(tmp_path):
    _write(tmp_path / "faq.json", [
        {"id": "q1", "text": "Como reinicio mi clave?", "category": "cuentas", "tags": ["login", "clave"], "votes": 3},
        {"id": "q2", "text": "Como pago?", "nested": {"no": "se indexa"}},
    ])
    a, b = JsonLoader(tmp_path).load()
    assert a.page_content == "Como reinicio mi clave?"
    assert a.metadata == {"source": "faq.json", "doc_id": "faq/q1", "category": "cuentas",
                          "record_id": "q1", "tags": ["login", "clave"], "votes": 3}
    assert b.metadata["category"] == "general" and "nested" not in b.metadata


def test_jsonl_y_id_por_posicion(tmp_path):
    (tmp_path / "datos").mkdir()
    (tmp_path / "datos" / "notas.jsonl").write_text('{"text": "uno"}\n\n{"text": "dos"}\n', encoding="utf-8")
    docs = JsonLoader(tmp_path).load()
    assert [d.metadata["doc_id"] for d in docs] == ["datos/notas/0", "datos/notas/1"]
    assert docs[0].metadata["category"] == "datos"


def test_json_claves_reservadas_no_se_sobrescriben(tmp_path):
    _write(tmp_path / "a.json", [{"id": 1, "text": "t", "doc_id": "hack", "chunk_id": "x", "source": "otro"}])
    meta = JsonLoader(tmp_path).load()[0].metadata
    assert meta["doc_id"] == "a/1" and meta["source"] == "a.json" and "chunk_id" not in meta


def test_json_claves_personalizadas(tmp_path):
    _write(tmp_path / "a.json", [{"slug": "x", "body": "contenido"}])
    doc = JsonLoader(tmp_path, text_key="body", id_key="slug").load()[0]
    assert doc.page_content == "contenido" and doc.metadata["doc_id"] == "a/x"


@pytest.mark.parametrize("data, mensaje", [
    ([{"id": 1}], "falta el campo de texto"),
    ([{"text": "   "}], "falta el campo de texto"),
    ([{"id": 1, "text": "a"}, {"id": 1, "text": "b"}], "id duplicado"),
    ({"text": "no es lista"}, "lista de objetos"),
    (["texto suelto"], "lista de objetos"),
])
def test_json_datos_invalidos_fallan_con_ubicacion(tmp_path, data, mensaje):
    _write(tmp_path / "mal.json", data)
    with pytest.raises(ValueError, match=mensaje):
        JsonLoader(tmp_path).load()


def test_json_error_indica_archivo_y_posicion(tmp_path):
    _write(tmp_path / "mal.json", [{"text": "ok"}, {"id": 2}])
    with pytest.raises(ValueError, match=r"mal\.json\[1\]"):
        JsonLoader(tmp_path).load()


# ---------- compuesto ----------
def test_default_loader_combina_formatos(tmp_path):
    (tmp_path / "doc.md").write_text("# Titulo\ntexto", encoding="utf-8")
    make_pdf(tmp_path / "manual.pdf", ["pagina"])
    _write(tmp_path / "faq.json", [{"id": "a", "text": "respuesta"}])
    docs = default_loader(tmp_path).load()
    assert sorted(d.metadata["source"] for d in docs) == ["doc.md", "faq.json", "manual.pdf"]
    assert {d.metadata["doc_id"] for d in docs} == {"doc", "manual", "faq/a"}


def test_compuesto_rechaza_doc_id_repetido_entre_archivos(tmp_path):
    (tmp_path / "guia.md").write_text("# G", encoding="utf-8")
    make_pdf(tmp_path / "guia.pdf", ["x"])
    with pytest.raises(ValueError, match="doc_id repetido 'guia'"):
        default_loader(tmp_path).load()


def test_compuesto_permite_paginas_del_mismo_archivo(tmp_path):
    make_pdf(tmp_path / "m.pdf", ["a", "b"])
    assert len(CompositeLoader([PdfLoader(tmp_path)]).load()) == 2


def test_markdown_sigue_ignorando_otros_formatos(tmp_path):
    make_pdf(tmp_path / "m.pdf", ["a"])
    _write(tmp_path / "f.json", [{"text": "t"}])
    assert MarkdownLoader(tmp_path).load() == []
