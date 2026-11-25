"""Loaders de documentos. Todos exponen `load() -> list[Document]` (protocolo `Loader`)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from langchain_core.documents import Document
from pypdf import PdfReader

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

def _category(rel: Path) -> str:
    """Primera carpeta del archivo ('general' si está en la raíz)."""
    return rel.parts[0] if len(rel.parts) > 1 else "general"


PAGE_SEPARATOR = "\n"            # un salto de línea más: con "\n\n" el splitter cortaría siempre en cada página
PAGE_MAP_KEY = "page_map"        # interno: [(offset, página)]; el chunking lo convierte en `page` y lo elimina


class PdfLoader:
    """Un Document por .pdf (el texto de todas sus páginas). Guarda dónde empieza cada página para que, tras
    el chunking, cada chunk (que puede cruzar páginas) lleve en `page` la página donde empieza (1-based).
    Las páginas sin texto (escaneadas) se omiten y se registran en `skipped_pages`."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.skipped_pages: list[tuple[str, int]] = []

    def load(self) -> list[Document]:
        self.skipped_pages = []
        docs = []
        for path in sorted(self.raw_dir.rglob("*.pdf")):
            rel = path.relative_to(self.raw_dir)
            reader = PdfReader(path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError(f"{rel.as_posix()} está cifrado con contraseña: no se puede indexar")
            title = (reader.metadata.title if reader.metadata else None) or ""
            texts, page_map, offset = [], [], 0
            for number, page in enumerate(reader.pages, start=1):
                text = self._clean(page.extract_text() or "")
                if not text:
                    self.skipped_pages.append((rel.as_posix(), number))
                    continue
                page_map.append((offset, number))
                texts.append(text)
                offset += len(text) + len(PAGE_SEPARATOR)
            if not texts:
                continue
            metadata = {"source": rel.as_posix(), "doc_id": rel.with_suffix("").as_posix(),
                        "category": _category(rel), "total_pages": len(reader.pages),
                        PAGE_MAP_KEY: page_map}
            if title:
                metadata["title"] = title
            docs.append(Document(page_content=PAGE_SEPARATOR.join(texts), metadata=metadata))
        return docs

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text.replace("\x00", ""))
        return re.sub(r"\n{3,}", "\n\n", text).strip()


RESERVED_KEYS = {"source", "doc_id", "chunk_id", "chunk_index", "text"}      # los fija la ingesta


def _is_filterable(value: Any) -> bool:
    """Tipos que Pinecone acepta como metadata: str, número, bool o lista de str."""
    return isinstance(value, (str, int, float, bool)) or (
        isinstance(value, list) and all(isinstance(v, str) for v in value))


class JsonLoader:
    """Registros de `.json` (lista de objetos) o `.jsonl` (un objeto por línea).
    Cada registro es un Document: `text_key` es el contenido; `category` y el resto de campos escalares
    (o listas de str, p. ej. `tags`) pasan a la metadata. doc_id = `<archivo>/<id>`."""

    def __init__(self, raw_dir: Path, *, text_key: str = "text", id_key: str = "id"):
        self.raw_dir, self.text_key, self.id_key = Path(raw_dir), text_key, id_key

    def load(self) -> list[Document]:
        docs = []
        for path in sorted([*self.raw_dir.rglob("*.json"), *self.raw_dir.rglob("*.jsonl")]):
            rel = path.relative_to(self.raw_dir)
            seen: set[str] = set()
            for index, record in enumerate(self._read(path)):
                where = f"{rel.as_posix()}[{index}]"
                text = record.get(self.text_key)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{where}: falta el campo de texto '{self.text_key}' (o está vacío)")
                record_id = str(record.get(self.id_key, index))
                if record_id in seen:
                    raise ValueError(f"{where}: id duplicado {record_id!r}")
                seen.add(record_id)
                category = record.get("category")
                metadata = {
                    **{k: v for k, v in record.items()
                       if k not in {self.text_key, self.id_key} | RESERVED_KEYS and _is_filterable(v)},
                    "source": rel.as_posix(),
                    "doc_id": f"{rel.with_suffix('').as_posix()}/{record_id}",
                    "category": category if isinstance(category, str) and category else _category(rel),
                    "record_id": record_id,
                }
                docs.append(Document(page_content=text.strip(), metadata=metadata))
        return docs

    @staticmethod
    def _read(path: Path) -> list[dict]:
        raw = path.read_text(encoding="utf-8-sig")
        records = ([json.loads(line) for line in raw.splitlines() if line.strip()]
                   if path.suffix == ".jsonl" else json.loads(raw))
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValueError(f"{path.name}: se esperaba una lista de objetos JSON")
        return records


class CompositeLoader:
    """Combina loaders y rechaza que dos archivos distintos produzcan el mismo `doc_id`."""

    def __init__(self, loaders: list[Loader]):
        self.loaders = loaders

    def load(self) -> list[Document]:
        docs, owner = [], {}
        for loader in self.loaders:
            for doc in loader.load():
                doc_id, source = doc.metadata["doc_id"], doc.metadata["source"]
                if owner.setdefault(doc_id, source) != source:
                    raise ValueError(f"doc_id repetido {doc_id!r}: {owner[doc_id]} y {source}")
                docs.append(doc)
        return docs


def default_loader(raw_dir: Path, code_dir: Path | None = None) -> CompositeLoader:
    """Markdown + PDF + JSON/JSONL de `raw_dir`; cada loader toma solo sus extensiones."""
    return CompositeLoader([MarkdownLoader(raw_dir, code_dir), PdfLoader(raw_dir), JsonLoader(raw_dir)])
