"""Infraestructura Pinecone: cliente, creación idempotente del índice Serverless e Index."""
from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from src.settings import Settings


def get_client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def get_index(settings: Settings, pc: Pinecone | None = None):
    return (pc or get_client(settings)).Index(settings.index_name)


def ensure_index(settings: Settings, pc: Pinecone | None = None, poll_seconds: float = 2.0) -> str:
    """Crea el índice si no existe; si existe, valida la dimensión. Espera a que esté listo."""
    pc = pc or get_client(settings)
    if settings.index_name in pc.list_indexes().names():
        dim = pc.describe_index(settings.index_name).dimension
        if dim != settings.embedding_dim:
            raise RuntimeError(
                f"Mismatch de dimensiones: el índice '{settings.index_name}' tiene {dim} y el modelo "
                f"produce {settings.embedding_dim}. Usá otro INDEX_NAME o borrá el índice."
            )
        print(f"[ok] Índice '{settings.index_name}' ya existe (dim={dim}).")
    else:
        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dim,
            metric=settings.metric,
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
        print(f"[ok] Índice '{settings.index_name}' creado (Serverless {settings.cloud}/{settings.region}).")
    while not pc.describe_index(settings.index_name).status["ready"]:
        time.sleep(poll_seconds)
    return settings.index_name
