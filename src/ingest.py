"""Pipeline de ingesta: Markdown -> chunks -> embeddings -> Pinecone (con metadata)."""
import os
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from src import config
from src.pinecone_setup import ensure_index

RAW_DIR = Path("data/raw")


def load_documents(raw_dir: Path = RAW_DIR) -> list[Document]:
    """Un Document por archivo .md. doc_id = ruta relativa sin extensión."""
    docs = []
    for path in sorted(raw_dir.rglob("*.md")):
        rel = path.relative_to(raw_dir)
        doc_id = rel.with_suffix("").as_posix()          # p.ej. tutorial/query-params
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\{\*.*?\*\}", "", text)            # includes de código de mkdocs
        docs.append(Document(
            page_content=text,
            metadata={"source": rel.as_posix(), "doc_id": doc_id, "category": category},
        ))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    section, last_doc = "", None
    for c in chunks:
        if c.metadata["doc_id"] != last_doc:
            section, last_doc = "", c.metadata["doc_id"]
        n = counters.get(c.metadata["doc_id"], 0)
        counters[c.metadata["doc_id"]] = n + 1
        c.metadata["chunk_index"] = n                       # posición del chunk en el documento
        heads = re.findall(r"^#{1,4}\s+(.*?)(?:\s*\{ #.*\})?\s*$", c.page_content, re.M)
        section = heads[0] if heads else section            # sin encabezado propio: hereda el anterior
        c.metadata["section"] = section
        c.metadata["chunk_id"] = f'{c.metadata["doc_id"]}#{n}'
    return chunks


def reset_namespace() -> None:
    """Deja el namespace como espejo de data/raw e invalida la caché BM25."""
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(config.INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=config.NAMESPACE)
    except Exception as exc:                      # namespace inexistente en la 1a ingesta
        if "not found" not in str(exc).lower():
            raise
    config.CORPUS_CACHE.unlink(missing_ok=True)


def ingest() -> int:
    config.require_keys()
    ensure_index()
    docs = load_documents()
    if not docs:
        sys.exit("No hay documentos en data/raw. Ver README (paso de descarga).")
    chunks = split_documents(docs)
    print(f"{len(docs)} documentos -> {len(chunks)} chunks")

    reset_namespace()
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    store = PineconeVectorStore(
        index_name=config.INDEX_NAME,
        embedding=embeddings,
        namespace=config.NAMESPACE,
        text_key="text",   # el texto vive en la metadata
    )
    # ids determinísticos -> reingestar es idempotente
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks], batch_size=100)
    print(f"[ok] {len(chunks)} chunks subidos al namespace '{config.NAMESPACE}'")
    return len(chunks)


if __name__ == "__main__":
    ingest()

