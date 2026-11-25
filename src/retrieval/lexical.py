"""Recuperación léxica (BM25). Aísla el import deprecado de langchain-community en un único lugar."""
from __future__ import annotations

import re
import warnings

from langchain_core.documents import Document

with warnings.catch_warnings():          # aviso de 'sunset' de langchain-community: solo se silencia aquí
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())      # 'Depends(' == 'depends'


def build_bm25(docs: list[Document], k: int) -> BM25Retriever:
    return BM25Retriever.from_documents(docs, k=k, preprocess_func=tokenize)
