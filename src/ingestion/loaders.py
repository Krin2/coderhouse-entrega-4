"""Loaders de documentos. Todos exponen `load() -> list[Document]` (protocolo `Loader`)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

# {* ../../docs_src/ruta/archivo.py ln[1:13] hl[2,5] *}  (include de mkdocs de FastAPI)
_INCLUDE = re.compile(r"\{\*\s*(\S+)(.*?)\*\}")
_LINES = re.compile(r"\bln\[([^\]]*)\]")
_FENCE_LANG = {".py": "python", ".html": "html", ".js": "javascript", ".json": "json", ".sh": "bash",
               ".toml": "toml", ".yml": "yaml", ".yaml": "yaml", ".md": "markdown"}


class Loader(Protocol):
    def load(self) -> list[Document]: ...


def select_lines(code: str, spec: str) -> str:
    """Aplica una especificación `ln[...]` (1-based, inclusiva): '1:13', ':5', '7:', '3' o varias con comas."""
    lines = code.splitlines()
    parts = []
    for rng in (r.strip() for r in spec.split(",")):
        if not rng:
            continue
        if ":" in rng:
            a, b = rng.split(":", 1)
            start, end = (int(a) if a else 1), (int(b) if b else len(lines))
        else:
            start = end = int(rng)
        parts.append("\n".join(lines[max(start, 1) - 1:end]))
    return "\n# ...\n".join(parts)


def inline_code(text: str, code_dir: Path | None) -> str:
    """Reemplaza los includes `{* ... *}` por bloques de código leídos de `code_dir`.
    Sin `code_dir` (o si el archivo no existe) el include se elimina. Un mismo fragmento se inserta una
    sola vez por documento: la documentación original lo repite con distintos resaltados."""
    root = Path(code_dir).resolve() if code_dir else None
    seen: set[tuple[str, str]] = set()

    def replace(match: re.Match) -> str:
        if root is None:
            return ""
        rel = match.group(1).split("docs_src/", 1)[-1]
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():      # fuera de code_dir o inexistente
            return ""
        lines_match = _LINES.search(match.group(2))
        key = (rel, lines_match.group(1) if lines_match else "")
        if key in seen:
            return ""
        seen.add(key)
        code = path.read_text(encoding="utf-8")
        if lines_match:
            code = select_lines(code, lines_match.group(1))
        lang = _FENCE_LANG.get(path.suffix, "")
        return f"\n```{lang}\n{code.rstrip()}\n```\n"

    return _INCLUDE.sub(replace, text)


class MarkdownLoader:
    """Un Document por archivo .md. `doc_id` = ruta relativa sin extensión;
    `category` = primera carpeta ('general' si el archivo está en la raíz).
    Con `code_dir` se incrusta el código referenciado por los includes de la documentación."""

    def __init__(self, raw_dir: Path, code_dir: Path | None = None):
        self.raw_dir = Path(raw_dir)
        self.code_dir = Path(code_dir) if code_dir else None

    def load(self) -> list[Document]:
        docs = []
        for path in sorted(self.raw_dir.rglob("*.md")):
            rel = path.relative_to(self.raw_dir)
            docs.append(Document(
                page_content=inline_code(path.read_text(encoding="utf-8"), self.code_dir),
                metadata={
                    "source": rel.as_posix(),
                    "doc_id": rel.with_suffix("").as_posix(),
                    "category": rel.parts[0] if len(rel.parts) > 1 else "general",
                },
            ))
        return docs
