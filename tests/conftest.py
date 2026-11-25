"""Fixtures: corpus sintético y vectorstore falso (sin red)."""
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_doc(chunk_id: str, text: str, category: str) -> Document:
    return Document(page_content=text, metadata={
        "chunk_id": chunk_id, "doc_id": chunk_id.split("#")[0], "category": category, "source": f"{chunk_id}.md"})


@pytest.fixture
def corpus():
    return [
        make_doc("tutorial/bg#0", "Use BackgroundTasks to run work after returning a response", "tutorial"),
        make_doc("tutorial/cors#0", "CORSMiddleware allow_origins configuration for browsers", "tutorial"),
        make_doc("advanced/ws#0", "WebSocketDisconnect is raised when the client disconnects", "advanced"),
        make_doc("advanced/events#0", "lifespan startup and shutdown events", "advanced"),
    ]


class FakeVectorStore:
    """Imita as_retriever(): devuelve docs del corpus filtrando por metadata, en orden fijo."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.calls = []

    def as_retriever(self, search_kwargs=None):
        store, kwargs = self, search_kwargs or {}
        store.calls.append(kwargs)

        class _R(BaseRetriever):
            def _get_relevant_documents(self, query, *, run_manager=None):
                cat = kwargs.get("filter", {}).get("category", {}).get("$eq")
                docs = [d for d in store.corpus if cat is None or d.metadata["category"] == cat]
                return docs[: kwargs.get("k", 4)]

        return _R()


@pytest.fixture
def fake_store(corpus):
    return FakeVectorStore(corpus)

def make_pdf(path, pages: list[str]) -> None:
    """Escribe un PDF mínimo y válido con una página por elemento (líneas separadas por \\n)."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        body = " T* ".join(f"({esc(line)}) Tj" for line in text.split("\n"))
        stream = f"BT /F1 12 Tf 14 TL 72 720 Td {body} ET".encode("latin-1")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>".encode())
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
