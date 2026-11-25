"""Chunking por tokens y enriquecimiento de metadata (chunk_id, chunk_index, section, page)."""
from __future__ import annotations

import re
from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import PAGE_MAP_KEY

_HEADING = re.compile(r"^#{1,4}\s+(.*?)(?:\s*\{ #.*\})?\s*$", re.M)   # quita anclas '{ #id }'


def extract_section(text: str, fallback: str = "") -> str:
    """Primer encabezado Markdown del texto, o `fallback` si no tiene."""
    headings = _HEADING.findall(text)
    return headings[0] if headings else fallback


def start_page(page_map: list[tuple[int, int]], char_index: int, fallback: int) -> int:
    """Página donde cae `char_index`, dado el mapa [(offset de inicio, página)]. Con índice inválido (-1)
    devuelve `fallback`."""
    if char_index < 0:
        return fallback
    return page_map[max(bisect_right([offset for offset, _ in page_map], char_index) - 1, 0)][1]


def split_documents(docs: list[Document], *, chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for doc in docs:
        # El mapa de páginas es una lista de tuplas (no es metadata válida de Pinecone): no pasa a los chunks.
        page_map = doc.metadata.get(PAGE_MAP_KEY)
        clean = Document(page_content=doc.page_content,
                         metadata={k: v for k, v in doc.metadata.items() if k != PAGE_MAP_KEY})
        parts = splitter.split_documents([clean])
        if page_map:
            last = page_map[0][1]
            for part in parts:
                last = part.metadata["page"] = start_page(page_map, part.metadata["start_index"], last)
        chunks.extend(parts)

    counters: dict[str, int] = {}
    section, current_doc = "", None
    for chunk in chunks:
        chunk.metadata.pop("start_index", None)
        doc_id = chunk.metadata["doc_id"]
        if doc_id != current_doc:
            current_doc, section = doc_id, ""
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1
        section = extract_section(chunk.page_content, fallback=section)   # sin encabezado: hereda el anterior
        chunk.metadata.update(chunk_index=index, chunk_id=f"{doc_id}#{index}")
        if section:                                    # PDF/JSON sin encabezados: no se guarda vacío
            chunk.metadata["section"] = section
    return chunks
