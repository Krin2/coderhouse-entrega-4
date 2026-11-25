"""Fábrica del PineconeVectorStore y del retriever semántico."""
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


def get_vectorstore() -> PineconeVectorStore:
    config.require_keys()
    return PineconeVectorStore(
        index_name=config.INDEX_NAME,
        embedding=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
        namespace=config.NAMESPACE,
        text_key="text",
    )


def get_semantic_retriever(k: int = config.TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    for d in get_semantic_retriever().invoke("How do I declare optional query parameters?"):
        print(d.metadata["chunk_id"], "|", d.page_content[:70].replace("\n", " "))
