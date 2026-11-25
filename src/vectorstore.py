"""Fábrica del PineconeVectorStore (el texto vive en la metadata, `text_key="text"`)."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.settings import Settings


def get_vectorstore(settings: Settings) -> PineconeVectorStore:
    settings.require_keys()
    return PineconeVectorStore(
        index_name=settings.index_name,
        embedding=OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key),
        namespace=settings.namespace,
        text_key="text",
        pinecone_api_key=settings.pinecone_api_key,
    )
