"""Configuración inmutable. No lee el entorno al importarse: usar `Settings.from_env()`."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # Credenciales (repr=False: nunca aparecen en logs ni tracebacks)
    pinecone_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)

    # Pinecone
    index_name: str = "rag-fastapi-docs"
    namespace: str = "docs-fastapi"
    cloud: str = "aws"
    region: str = "us-east-1"
    metric: str = "cosine"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Chunking (en tokens) y recuperación
    chunk_size: int = 600
    chunk_overlap: int = 80
    top_k: int = 5

    # Rutas
    raw_dir: Path = Path("data/raw")
    corpus_cache: Path = Path("data/cache/corpus.jsonl")
    golden_set: Path = Path("eval/golden_set.json")
    report_dir: Path = Path("reports")

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        load_dotenv()
        env = os.environ.get
        base = cls(
            pinecone_api_key=env("PINECONE_API_KEY", ""),
            openai_api_key=env("OPENAI_API_KEY", ""),
            index_name=env("INDEX_NAME", cls.index_name),
            namespace=env("NAMESPACE", cls.namespace),
            cloud=env("PINECONE_CLOUD", cls.cloud),
            region=env("PINECONE_REGION", cls.region),
        )
        return replace(base, **overrides) if overrides else base

    def require_keys(self) -> None:
        missing = [name for name, value in (
            ("PINECONE_API_KEY", self.pinecone_api_key), ("OPENAI_API_KEY", self.openai_api_key)) if not value]
        if missing:
            raise RuntimeError(f"Faltan variables en .env: {', '.join(missing)}")
