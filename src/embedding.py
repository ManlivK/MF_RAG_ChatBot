"""Stage 3 - Embedding with all-MiniLM-L6-v2 (ARCHITECTURE.md section 4.3).

Documents are embedded with an asymmetric prefix (``config.DOC_EMBED_PREFIX``);
queries are embedded **unprefixed**. The asymmetry is intentional and is part of the
embedding fingerprint, so changing one side without the other marks the persisted
index stale instead of silently comparing vectors from two different spaces.
"""

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
    """Return the process-wide MiniLM embedder.

    Constructed once per process (Streamlit reruns the script on every interaction,
    and loading the model each time would blow the NFR2 latency budget).

    Returns:
        A ``HuggingFaceEmbeddings`` configured for L2-normalised, CPU-side encoding
        of 384-dimensional vectors.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
            # ARCH section 4.3 sets encode_batch_size = 32. In this version of
            # langchain-huggingface the batch size lives in encode_kwargs rather
            # than a top-level embed_batch_size field.
            "batch_size": config.EMBED_BATCH_SIZE,
        },
        model_kwargs={"device": config.EMBED_DEVICE},
        show_progress=False,
    )


def embedding_fingerprint() -> str:
    """Short hash identifying the exact document-side embedding space.

    The fingerprint covers everything that changes the stored vectors: the model
    name, the document prefix and the normalisation flag. The device is **not**
    part of the hash - CPU and GPU produce identical vectors for this model, so
    folding the device in would only ever mark a perfectly good index stale.
    The concrete device is recorded separately in ``index_meta.json``.

    Returns:
        The first 12 hex characters of ``sha256(model|prefix|normalize)``.
    """
    material = "|".join(
        (
            config.EMBED_MODEL,
            config.DOC_EMBED_PREFIX,
            "normalize=True",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts on the corpus side.

    Args:
        texts: Raw chunk texts as stored in ``data/chunks/chunks.jsonl``.

    Returns:
        One L2-normalised 384-dimensional vector per input, in input order. The
        document prefix is prepended to every text before encoding.
    """
    if not texts:
        return []
    prefixed = [f"{config.DOC_EMBED_PREFIX}{text}" for text in texts]
    return get_embeddings().embed_documents(prefixed)


def embed_query(text: str) -> list[float]:
    """Embed a query string on the query side.

    Args:
        text: The expanded query from ``QueryPlan.expanded``.

    Returns:
        One L2-normalised 384-dimensional vector. **No** document prefix is applied -
        see the module docstring.
    """
    return get_embeddings().embed_query(text)