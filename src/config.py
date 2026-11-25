"""Configuración central: carga .env y expone constantes validadas."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
METRIC = "cosine"

INDEX_NAME = os.getenv("INDEX_NAME", "rag-fastapi-docs")
NAMESPACE = os.getenv("NAMESPACE", "docs-fastapi")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

CHUNK_SIZE = 600      # tokens
CHUNK_OVERLAP = 80    # tokens
TOP_K = 5

CORPUS_CACHE = Path("data/cache/corpus.jsonl")   # corpus BM25 (se invalida solo)


def require_keys() -> None:
    missing = [k for k in ("PINECONE_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Faltan variables en .env: {', '.join(missing)}")
