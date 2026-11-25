import json

import pytest

from src.ingestion.chunking import split_documents, start_page
from src.ingestion.loaders import CompositeLoader, JsonLoader, MarkdownLoader, PdfLoader, default_loader
from tests.conftest import make_pdf


# ---------- PDF ----------
def _chunks(pdf_dir, size=600, overlap=0):
    return split_documents(PdfLoader(pdf_dir).load(), chunk_size=size, chunk_overlap=overlap)


def test_pdf_un_documento_por_archivo_con_mapa_de_paginas(tmp_path):
    (tmp_path / "guias").mkdir()
    make_pdf(tmp_path / "guias" / "manual.pdf", ["Capitulo uno\nsobre cache", "Capitulo dos", "Capitulo tres"])
    (doc,) = PdfLoader(tmp_path).load()
    assert "Capitulo uno" in doc.page_content and "Capitulo tres" in doc.page_content
    assert [p for _, p in doc.metadata["page_map"]] == [1, 2, 3]
    assert {k: v for k, v in doc.metadata.items() if k != "page_map"} == {
        "source": "guias/manual.pdf", "doc_id": "guias/manual", "category": "guias", "total_pages": 3}


def test_pdf_offsets_apuntan_al_inicio_real_de_cada_pagina(tmp_path):
    make_pdf(tmp_path / "m.pdf", ["alpha uno", "beta dos", "gamma tres"])
    (doc,) = PdfLoader(tmp_path).load()
    for (offset, _), word in zip(doc.metadata["page_map"], ("alpha", "beta", "gamma")):
        assert doc.page_content[offset:].startswith(word)


def test_pdf_chunk_que_cruza_paginas_usa_la_pagina_de_inicio(tmp_path):
    make_pdf(tmp_path / "doc.pdf", ["alpha " * 8, "beta " * 8, "gamma " * 8])
    chunks = _chunks(tmp_path, size=40)
    crossing = [c for c in chunks if "alpha" in c.page_content and "beta" in c.page_content]
    assert crossing and crossing[0].metadata["page"] == 1              # empieza en 1 aunque llegue a la 2
    assert all(isinstance(c.metadata["page"], int) for c in chunks)


def test_pdf_multilinea_realista_cruza_paginas_y_conserva_la_de_inicio(tmp_path):
    def page(tag):
        return "\n".join(" ".join(f"{tag}_{line}_{w}" for w in range(10)) for line in range(14))

    make_pdf(tmp_path / "informe.pdf", [page("p1"), page("p2"), page("p3")])
    chunks = _chunks(tmp_path, size=200, overlap=30)
    crossing = [c for c in chunks if {"p1_", "p2_"} <= {w[:3] for w in c.page_content.split()}]
    assert crossing and all(c.metadata["page"] == 1 for c in crossing)       # mezcla p1 y p2, página = 1
    only_p2 = [c for c in chunks if all(w.startswith("p2_") for w in c.page_content.split())]
    assert only_p2 and all(c.metadata["page"] == 2 for c in only_p2)


def test_pdf_paginas_monotonas_y_cubren_el_documento(tmp_path):
    make_pdf(tmp_path / "doc.pdf", [f"pagina{i} " * 30 for i in range(1, 6)])
    pages = [c.metadata["page"] for c in _chunks(tmp_path, size=25)]
    assert pages == sorted(pages) and pages[0] == 1 and pages[-1] == 5


def test_pdf_chunks_no_llevan_metadata_interna(tmp_path):
    make_pdf(tmp_path / "doc.pdf", ["uno", "dos"])
    for chunk in _chunks(tmp_path):
        assert not {"page_map", "start_index"} & chunk.metadata.keys()
        assert chunk.metadata["total_pages"] == 2 and "section" not in chunk.metadata


def test_pdf_ids_de_chunk_continuos_a_traves_de_paginas(tmp_path):
    make_pdf(tmp_path / "doc.pdf", ["alpha " * 30, "beta " * 30])
    chunks = _chunks(tmp_path, size=20)
    assert [c.metadata["chunk_id"] for c in chunks] == [f"doc#{i}" for i in range(len(chunks))]


def test_split_documents_no_muta_la_entrada(tmp_path):
    make_pdf(tmp_path / "doc.pdf", ["uno", "dos"])
    docs = PdfLoader(tmp_path).load()
    split_documents(docs, chunk_size=600, chunk_overlap=0)
    assert "page_map" in docs[0].metadata
    split_documents(docs, chunk_size=600, chunk_overlap=0)           # segunda vez: mismas páginas
    assert split_documents(docs, chunk_size=600, chunk_overlap=0)[0].metadata["page"] == 1


def test_pdf_omite_paginas_sin_texto_y_las_registra(tmp_path):
    make_pdf(tmp_path / "scan.pdf", ["hola", "", "adios"])
    loader = PdfLoader(tmp_path)
    (doc,) = loader.load()
    assert [p for _, p in doc.metadata["page_map"]] == [1, 3]
    assert loader.skipped_pages == [("scan.pdf", 2)]


def test_pdf_sin_ningun_texto_no_genera_documento(tmp_path):
    make_pdf(tmp_path / "vacio.pdf", ["", ""])
    loader = PdfLoader(tmp_path)
    assert loader.load() == [] and len(loader.skipped_pages) == 2


def test_pdf_categoria_general_en_la_raiz(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["x"])
    assert PdfLoader(tmp_path).load()[0].metadata["category"] == "general"


def test_pdf_corrupto_falla_de_forma_explicita(tmp_path):
    (tmp_path / "roto.pdf").write_bytes(b"esto no es un pdf")
    with pytest.raises(Exception):
        PdfLoader(tmp_path).load()


def test_start_page():
    page_map = [(0, 1), (100, 2), (250, 4)]
    assert [start_page(page_map, i, 9) for i in (0, 99, 100, 249, 250, 999)] == [1, 1, 2, 2, 4, 4]
    assert start_page(page_map, -1, 9) == 9                           # índice no encontrado => fallback

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


def test_compuesto_mismo_archivo_dos_veces_no_es_conflicto(tmp_path):
    make_pdf(tmp_path / "m.pdf", ["a", "b"])
    assert len(CompositeLoader([PdfLoader(tmp_path), PdfLoader(tmp_path)]).load()) == 2


def test_markdown_sigue_ignorando_otros_formatos(tmp_path):
    make_pdf(tmp_path / "m.pdf", ["a"])
    _write(tmp_path / "f.json", [{"text": "t"}])
    assert MarkdownLoader(tmp_path).load() == []
