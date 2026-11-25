"""Loaders de documentos. Todos exponen `load() -> list[Document]` (protocolo `Loader`)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

_MKDOCS_INCLUDE = re.compile(r"\{\*.*?\*\}")        # includes de código de mkdocs


class Loader(Protocol):
    def load(self) -> list[Document]: ...


class MarkdownLoader:
    """Un Document por archivo .md. `doc_id` = ruta relativa sin extensión;
    `category` = primera carpeta ('general' si el archivo está en la raíz)."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def load(self) -> list[Document]:
        docs = []
        for path in sorted(self.raw_dir.rglob("*.md")):
            rel = path.relative_to(self.raw_dir)
            docs.append(Document(
                page_content=_MKDOCS_INCLUDE.sub("", path.read_text(encoding="utf-8")),
                metadata={
                    "source": rel.as_posix(),
                    "doc_id": rel.with_suffix("").as_posix(),
                    "category": rel.parts[0] if len(rel.parts) > 1 else "general",
                },
            ))
        return docs
