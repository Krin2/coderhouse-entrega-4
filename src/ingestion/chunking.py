"""Chunking por tokens y enriquecimiento de metadata (chunk_id, chunk_index, section, page)."""
from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable

import tiktoken
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


def count_tokens(text: str) -> int:
    """Tokens cl100k_base (el mismo conteo que usa el splitter y text-embedding-3-small)."""
    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def locate_chunks(parts: list[Document], text: str) -> list[Document]:
    """Anota `start_index` (posición del chunk en `text`). No se usa `add_start_index` del splitter: busca
    desde `inicio_anterior + largo_anterior - chunk_overlap` tratando el solapamiento (en tokens) como
    caracteres, y devuelve -1 cuando el solapamiento real es mayor. Aquí cada chunk empieza después del
    anterior, así que se busca desde `inicio_anterior + 1`. Si no se encuentra queda en -1."""
    previous = -1
    for part in parts:
        index = text.find(part.page_content, previous + 1)
        part.metadata["start_index"] = index
        if index >= 0:
            previous = index
    return parts


def merge_small_chunks(parts: list[Document], text: str, *, min_tokens: int, max_tokens: int,
                       count: Callable[[str], int] = count_tokens) -> list[Document]:
    """Fusiona cada chunk de menos de `min_tokens` con su vecino (recorrido de izquierda a derecha, así
    cubre colas y cabezas pequeñas) mientras el resultado no pase de `max_tokens`. El chunk fusionado es
    el tramo contiguo del texto original, así el solapamiento del splitter no queda duplicado."""
    starts = [p.metadata.get("start_index", -1) for p in parts]
    if min_tokens <= 0 or len(parts) < 2 or min(starts) < 0:
        return parts
    merged: list[tuple[int, int, Document]] = []
    for part, start in zip(parts, starts):
        end = start + len(part.page_content)
        if merged:
            prev_start, prev_end, prev = merged[-1]
            small = count(text[prev_start:prev_end]) < min_tokens or count(part.page_content) < min_tokens
            if small and count(text[prev_start:end]) <= max_tokens:
                merged[-1] = (prev_start, max(prev_end, end), prev)
                continue
        merged.append((start, end, part))
    return [Document(page_content=text[s:e], metadata=dict(p.metadata)) for s, e, p in merged]


def split_documents(docs: list[Document], *, chunk_size: int, chunk_overlap: int,
                    min_tokens: int = 0, max_tokens: int | None = None) -> list[Document]:
    """`min_tokens`/`max_tokens` activan la fusión de chunks pequeños (por defecto, desactivada)."""
    max_tokens = max_tokens or chunk_size
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in docs:
        # El mapa de páginas es una lista de tuplas (no es metadata válida de Pinecone): no pasa a los chunks.
        page_map = doc.metadata.get(PAGE_MAP_KEY)
        clean = Document(page_content=doc.page_content,
                         metadata={k: v for k, v in doc.metadata.items() if k != PAGE_MAP_KEY})
        parts = locate_chunks(splitter.split_documents([clean]), clean.page_content)
        parts = merge_small_chunks(parts, clean.page_content, min_tokens=min_tokens, max_tokens=max_tokens)
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
