"""Chunking por tokens y enriquecimiento de metadata (chunk_id, chunk_index, section)."""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING = re.compile(r"^#{1,4}\s+(.*?)(?:\s*\{ #.*\})?\s*$", re.M)   # quita anclas '{ #id }'


def extract_section(text: str, fallback: str = "") -> str:
    """Primer encabezado Markdown del texto, o `fallback` si no tiene."""
    headings = _HEADING.findall(text)
    return headings[0] if headings else fallback


def split_documents(docs: list[Document], *, chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    section, current_doc = "", None
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        if doc_id != current_doc:
            current_doc, section = doc_id, ""
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1
        section = extract_section(chunk.page_content, fallback=section)   # sin encabezado: hereda el anterior
        chunk.metadata.update(chunk_index=index, chunk_id=f"{doc_id}#{index}", section=section)
    return chunks
