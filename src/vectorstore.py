"""Stage 4 - Vector store (ARCHITECTURE.md section 4.4, section 5.1).

The collection holds *pre-computed* MiniLM vectors, so it is created with
``embedding_function=None``. That is the single most important line in this file:
without it Chroma loads its own default embedder and the collection silently ends
up comparing vectors from two different models.

Everything the index needs to be reproducible - model, fingerprint, chunking
parameters, distance - is written to ``data/index_meta.json`` next to the
collection, and checked on every retrieval call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src import config
from src.embedding import embed_documents, embedding_fingerprint

__all__ = [
    "COLLECTION_NAME",
    "IndexStaleError",
    "get_client",
    "get_collection",
    "reset_collection_cache",
    "load_chunks",
    "rebuild_index",
    "upsert_chunks",
    "write_index_meta",
    "read_index_meta",
    "is_index_stale",
    "get_stats",
    "run_all",
    "main",
]

#: Chroma collection name. The canonical value lives in ``config.CHROMA_COLLECTION``
#: (ARCH section 6: no magic strings in stage modules); this alias keeps the
#: documented module-level name available to importers.
COLLECTION_NAME: str = config.CHROMA_COLLECTION

#: Metadata keys mirrored from the chunk record (ARCH section 4.4 collection contract).
METADATA_KEYS: tuple[str, ...] = (
    "chunk_id",
    "scheme_key",
    "scheme_name",
    "category",
    "section",
    "label",
    "source_url",
    "source_title",
    "fetched_at",
    "chunk_index",
    "content_hash",
)


class IndexStaleError(RuntimeError):
    """The persisted collection was built by a different embedding space.

    Raised instead of returning wrong-but-plausible results (ARCH section 4.6
    step 1: never silently compare vectors from different models).
    """


# --------------------------------------------------------------------------- #
# Client / collection lifecycle
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """Return the process-wide persistent Chroma client.

    Returns:
        A ``chromadb.PersistentClient`` rooted at ``data/chroma``.
    """
    chroma_dir: Path = config.PATHS["CHROMA_DIR"]
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    """Return the ``mf_faq_hdfc`` collection, creating it if absent.

    The ``lru_cache`` matters for the UI: Streamlit reruns the script on every
    interaction, and re-opening the collection each time is both slow and, after
    a rebuild, a handle onto a deleted collection. Call
    :func:`reset_collection_cache` after any rebuild.

    Returns:
        The Chroma collection with ``embedding_function=None``.
    """
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE},
        embedding_function=None,
    )


def reset_collection_cache() -> None:
    """Drop the memoised client/collection handles after a destructive rebuild."""
    get_collection.cache_clear()
    get_client.cache_clear()


# --------------------------------------------------------------------------- #
# Chunk loading
# --------------------------------------------------------------------------- #


def load_chunks() -> list[dict[str, Any]]:
    """Read every chunk record from ``data/chunks/chunks.jsonl``.

    Returns:
        The parsed chunk records in file order. An empty list when the file is
        absent, which callers treat as "nothing to index".
    """
    path: Path = config.PATHS["CHUNKS_FILE"]
    if not path.exists():
        return []
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt artifact
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
    return chunks


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Project a chunk record onto the Chroma metadata contract.

    Chroma rejects ``None`` values and non-scalar types, so every key is coerced
    to a primitive here rather than at the call site.
    """
    metadata: dict[str, Any] = {}
    for key in METADATA_KEYS:
        value = chunk.get(key)
        metadata[key] = "" if value is None else value
    metadata["chunk_index"] = int(chunk.get("chunk_index") or 0)
    return metadata


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _create_collection() -> Collection:
    """Create a fresh collection with cosine distance and no embedder."""
    return get_client().create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE},
        embedding_function=None,
    )


def rebuild_index(chunks: list[dict[str, Any]] | None = None) -> int:
    """Drop the collection, recreate it and insert every chunk.

    Chunk ids are positional (``scheme_key::0007``), so a re-ingest that changes
    earlier content shifts ids. A full delete-and-recreate is therefore the only
    correct rebuild path (ARCH section 4.4, "Idempotency").

    Args:
        chunks: Chunk records. Defaults to everything in ``chunks.jsonl``.

    Returns:
        The number of chunks written.
    """
    records = load_chunks() if chunks is None else chunks
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - absent collection is the normal first-run case
        pass
    reset_collection_cache()
    collection = _create_collection()
    written = 0
    if records:
        collection.upsert(
            ids=[record["chunk_id"] for record in records],
            documents=[record["text"] for record in records],
            embeddings=embed_documents([record["text"] for record in records]),
            metadatas=[_chunk_metadata(record) for record in records],
        )
        written = len(records)
    write_index_meta(_index_stats(records, collection))
    reset_collection_cache()
    return written


def upsert_chunks(chunks: list[dict[str, Any]]) -> int:
    """Insert or update chunks by ``chunk_id``, leaving untouched ones alone.

    The idempotent path for a partial refresh: ``content_hash`` lets a caller skip
    chunks whose text has not changed.

    Args:
        chunks: Chunk records to write.

    Returns:
        The number of chunks submitted to Chroma.
    """
    if not chunks:
        return 0
    collection = get_collection()
    collection.upsert(
        ids=[record["chunk_id"] for record in chunks],
        documents=[record["text"] for record in chunks],
        embeddings=embed_documents([record["text"] for record in chunks]),
        metadatas=[_chunk_metadata(record) for record in chunks],
    )
    write_index_meta(_index_stats(load_chunks(), collection))
    return len(chunks)


def _index_stats(records: list[dict[str, Any]], collection: Collection) -> dict[str, Any]:
    """Assemble the ``data/index_meta.json`` payload (ARCH section 5.1)."""
    doc_count = len({record.get("scheme_key") for record in records if record.get("scheme_key")})
    chunk_count = collection.count() if records else 0
    return {
        "collection": COLLECTION_NAME,
        "embedding_model": config.EMBED_MODEL,
        "embedding_fingerprint": embedding_fingerprint(),
        "embedding_dim": config.EMBED_DIM,
        "embedding_device": config.EMBED_DEVICE,
        "doc_embed_prefix": config.DOC_EMBED_PREFIX,
        "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "chunk_strategy": config.CHUNK_STRATEGY,
        "distance": config.CHROMA_DISTANCE_SPACE,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": config.GENERATOR_MODEL,
    }


def write_index_meta(stats: dict[str, Any]) -> None:
    """Persist the index fingerprint and build parameters."""
    path: Path = config.PATHS["INDEX_META_FILE"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def read_index_meta() -> dict[str, Any] | None:
    """Read ``data/index_meta.json``.

    Returns:
        The parsed metadata, or ``None`` when the file is missing or unreadable.
    """
    path: Path = config.PATHS["INDEX_META_FILE"]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_index_stale() -> tuple[bool, str]:
    """Report whether the persisted index can still be searched.

    Returns:
        ``(True, reason)`` when the metadata is missing or was written by a
        different embedding space, otherwise ``(False, "ok")``.
    """
    meta = read_index_meta()
    if not meta:
        return True, "missing meta"
    current = embedding_fingerprint()
    if meta.get("embedding_fingerprint") != current:
        return True, "fingerprint mismatch"
    return False, "ok"


def get_stats() -> dict[str, Any]:
    """Return the sidebar stats (PRD FR12).

    Falls back to a live ``count()`` so the sidebar still works before the first
    rebuild has written the metadata file.
    """
    meta = read_index_meta()
    if meta:
        return meta
    try:
        return {"collection": COLLECTION_NAME, "chunk_count": get_collection().count()}
    except Exception:  # noqa: BLE001 - sidebar must never crash the app
        return {"collection": COLLECTION_NAME, "chunk_count": 0}


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def run_all(rebuild: bool = True) -> int:
    """``run_index.py --stage index`` entry point.

    Args:
        rebuild: Drop and recreate the collection (default, the only safe path
            because chunk ids are positional). ``False`` upserts instead.

    Returns:
        The number of chunks written, which the CLI treats as success.
    """
    records = load_chunks()
    if not records:
        print("  no chunks found - run --stage chunk first")
        return 0
    written = rebuild_index(records) if rebuild else upsert_chunks(records)
    stale, reason = is_index_stale()
    print(f"  collection   : {COLLECTION_NAME} ({config.CHROMA_DISTANCE_SPACE})")
    print(f"  chunks       : {written}")
    print(f"  fingerprint  : {embedding_fingerprint()}")
    print(f"  stale        : {stale} ({reason})")
    return written


def main(rebuild: bool = True) -> int:
    """Alias of :func:`run_all` so the stage dispatcher can call either."""
    return run_all(rebuild=rebuild)