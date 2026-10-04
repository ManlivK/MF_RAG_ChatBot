"""Stage 3 - Embeddings with MiniLM (ARCHITECTURE.md section 4.3)."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src import config

__all__ = [
    "get_embeddings",
    "embedding_fingerprint",
    "embed_documents",
    "embed_query",
]


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    model = getattr(config, 'EMBED_MODEL', None) or getattr(config, 'EMBED_CONFIG_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    kwargs = {
        'model_name': model,
        'encode_kwargs': {"normalize_embeddings": True},
        'model_kwargs': {"device": config.EMBED_DEVICE},
        'show_progress': True,
    }
    return HuggingFaceEmbeddings(**kwargs)


def embedding_fingerprint() -> str:
    model = getattr(config, 'EMBED_MODEL', None) or getattr(config, 'EMBED_CONFIG_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    doc_prefix = config.DOC_EMBED_PREFIX
    normalize = True
    device_class = "cpu" if config.EMBED_DEVICE == "cpu" else "gpu"
    h = hashlib.sha256(f"{model}|{doc_prefix}|{normalize}|{device_class}".encode()).hexdigest()
    return h[:12]


def embed_documents(texts: list[str]) -> list[list[float]]:
    emb = get_embeddings()
    prefixed = [f"{config.DOC_EMBED_PREFIX}{t}" for t in texts]
    return emb.embed_documents(prefixed)


def embed_query(text: str) -> list[float]:
    emb = get_embeddings()
    return emb.embed_query(text)