"""Crea el índice Serverless si no existe y valida su dimensión."""
import os
import time
from pinecone import Pinecone, ServerlessSpec
from src import config


def ensure_index() -> str:
    config.require_keys()
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    names = pc.list_indexes().names()
    if config.INDEX_NAME in names:
        desc = pc.describe_index(config.INDEX_NAME)
        if desc.dimension != config.EMBEDDING_DIM:
            raise RuntimeError(
                f"Mismatch de dimensiones: el índice '{config.INDEX_NAME}' tiene "
                f"{desc.dimension} y el modelo produce {config.EMBEDDING_DIM}. "
                "Usá otro INDEX_NAME o borrá el índice."
            )
        print(f"[ok] Índice '{config.INDEX_NAME}' ya existe (dim={desc.dimension}).")
    else:
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric=config.METRIC,
            spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
        )
        print(f"[ok] Índice '{config.INDEX_NAME}' creado (Serverless {config.CLOUD}/{config.REGION}).")
    while not pc.describe_index(config.INDEX_NAME).status["ready"]:   # espera a que esté listo
        time.sleep(2)
    return config.INDEX_NAME


if __name__ == "__main__":
    ensure_index()
