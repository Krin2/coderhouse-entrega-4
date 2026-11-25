from langchain_core.documents import Document

from src.ingestion.chunking import (
    count_tokens,
    locate_chunks,
    merge_small_chunks,
    split_documents,
)


def _doc(text, doc_id="d"):
    return Document(page_content=text, metadata={"doc_id": doc_id, "source": f"{doc_id}.md", "category": "general"})


def _sections(*sizes):
    """Markdown con una sección por tamaño aproximado en tokens (' word' = 1 token)."""
    return "\n\n".join(f"## S{i}\n\n" + "word " * n for i, n in enumerate(sizes))


# ---------- ubicación de chunks ----------
def test_locate_chunks_con_solapamiento_largo_no_pierde_ninguno():
    # el solapamiento (80 tokens) son más caracteres que 80: add_start_index de LangChain devolvía -1
    text = " ".join(f"palabra{i}" for i in range(3000))
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=200, chunk_overlap=80)
    parts = locate_chunks(splitter.split_documents([_doc(text)]), text)
    assert len(parts) > 5
    assert all(p.metadata["start_index"] >= 0 for p in parts)
    assert all(text[p.metadata["start_index"]:].startswith(p.page_content) for p in parts)
    starts = [p.metadata["start_index"] for p in parts]
    assert starts == sorted(set(starts))                                  # estrictamente crecientes


def test_locate_chunks_no_encontrado_queda_en_menos_uno():
    parts = locate_chunks([_doc("texto que no está")], "otro contenido")
    assert parts[0].metadata["start_index"] == -1


# ---------- fusión de chunks pequeños ----------
def test_sin_minimo_no_fusiona():
    chunks = split_documents([_doc(_sections(580, 30, 580))], chunk_size=600, chunk_overlap=0)
    assert len(chunks) == 3 and min(count_tokens(c.page_content) for c in chunks) < 60


def test_chunk_pequeno_del_medio_se_fusiona_con_su_vecino():
    chunks = split_documents([_doc(_sections(580, 30, 580))], chunk_size=600, chunk_overlap=0,
                             min_tokens=200, max_tokens=800)
    sizes = [count_tokens(c.page_content) for c in chunks]
    assert len(chunks) == 2 and min(sizes) >= 200 and max(sizes) <= 800
    assert "## S1" in chunks[0].page_content or "## S1" in chunks[1].page_content   # nada se perdió


def test_chunk_pequeno_al_inicio_y_al_final_se_fusionan():
    chunks = split_documents([_doc(_sections(30, 580, 580, 30))], chunk_size=600, chunk_overlap=0,
                             min_tokens=200, max_tokens=800)
    sizes = [count_tokens(c.page_content) for c in chunks]
    assert len(chunks) == 2 and min(sizes) >= 200
    assert chunks[0].page_content.startswith("## S0") and "## S3" in chunks[-1].page_content


def test_fusion_respeta_el_maximo():
    text = _sections(590, 150, 590)                  # 590+150 = 740 > 700: no puede fusionarse
    chunks = split_documents([_doc(text)], chunk_size=600, chunk_overlap=0, min_tokens=200, max_tokens=700)
    assert max(count_tokens(c.page_content) for c in chunks) <= 700
    assert len(chunks) == 3


def test_fusion_no_duplica_el_solapamiento():
    body = " ".join(f"unico{i}" for i in range(1500))              # una sola sección larga => hay solapamiento
    plain = split_documents([_doc(body)], chunk_size=300, chunk_overlap=80)
    merged = split_documents([_doc(body)], chunk_size=300, chunk_overlap=80, min_tokens=250, max_tokens=700)
    for chunk in merged:
        words = chunk.page_content.split()
        assert len(words) == len(set(words))                       # ninguna palabra repetida dentro del chunk
    assert len(merged) <= len(plain)


def test_documento_de_un_solo_chunk_pequeno_queda_igual():
    chunks = split_documents([_doc("# Titulo\n\ncorto")], chunk_size=600, chunk_overlap=0,
                             min_tokens=200, max_tokens=800)
    assert len(chunks) == 1 and chunks[0].metadata["chunk_id"] == "d#0"


def test_ids_e_indices_continuos_tras_fusionar():
    chunks = split_documents([_doc(_sections(580, 30, 580, 30, 580))], chunk_size=600, chunk_overlap=0,
                             min_tokens=200, max_tokens=800)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert [c.metadata["chunk_id"] for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert not any("start_index" in c.metadata for c in chunks)


def test_la_seccion_de_un_chunk_fusionado_es_su_primer_encabezado():
    chunks = split_documents([_doc(_sections(580, 30, 580))], chunk_size=600, chunk_overlap=0,
                             min_tokens=200, max_tokens=800)
    assert chunks[0].metadata["section"] == "S0"


def test_merge_sin_posiciones_conocidas_no_toca_nada():
    parts = [_doc("a"), _doc("b")]
    for p in parts:
        p.metadata["start_index"] = -1
    assert merge_small_chunks(parts, "ab", min_tokens=200, max_tokens=800) is parts


def test_la_pagina_de_un_chunk_fusionado_es_la_de_su_inicio():
    from src.ingestion.loaders import PAGE_MAP_KEY

    def page(tag):
        return "\n".join(f"{tag} " + "word " * 12 for _ in range(14))

    pages = [page("p1"), page("p2"), page("p3")]
    text, page_map, offset = "\n".join(pages), [], 0
    for number, p in enumerate(pages, start=1):
        page_map.append((offset, number))
        offset += len(p) + 1
    doc = Document(page_content=text, metadata={"doc_id": "m", "source": "m.pdf", PAGE_MAP_KEY: page_map})
    for chunk in split_documents([doc], chunk_size=150, chunk_overlap=0, min_tokens=100, max_tokens=300):
        first = chunk.page_content.split()[0]
        assert chunk.metadata["page"] == int(first[1])      # un chunk que arranca en 'p2 ...' está en la página 2
