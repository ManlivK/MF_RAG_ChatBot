"""Phase 4 - retrieval."""
from __future__ import annotations

import json

from src import config
from src.embedding import embed_query, embedding_fingerprint
from src.vectorstore import get_collection, read_index_meta, IndexStaleError
from src.types import QueryPlan, RetrievedChunk, RetrievalResult


__all__ = ["retrieve", "format_context"]


def _check_stale():
    meta = read_index_meta()
    if not meta:
        raise IndexStaleError("index meta missing")
    if meta.get("embedding_fingerprint") != embedding_fingerprint():
        raise IndexStaleError("embedding fingerprint mismatch")


def retrieve(plan: QueryPlan) -> RetrievalResult:
    _check_stale()
    qvec = embed_query(plan.expanded)
    collection = get_collection()
    where = None
    if plan.scheme_key:
        where = {"scheme_key": plan.scheme_key}
    results = collection.query(
        query_embeddings=[qvec],
        n_results=config.RETRIEVE_K,
        where=where,
    )
    chunks = []
    dropped_low = 0
    dropped_scheme = 0
    distances = results.get("distances", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for i, (d, m) in enumerate(zip(distances, metas)):
        score = 1.0 - d
        if score < config.SIMILARITY_FLOOR:
            dropped_low += 1
            continue
        chunk = {
            "chunk_id": m.get("chunk_id"),
            "text": results.get("documents", [[]])[0][i] if len(results.get("documents", [[]])[0]) > i else "",
            "scheme_key": m.get("scheme_key"),
            "scheme_name": m.get("scheme_name"),
            "category": m.get("category"),
            "section": m.get("section"),
            "label": m.get("label"),
            "source_url": m.get("source_url"),
            "source_title": m.get("source_title"),
            "fetched_at": m.get("fetched_at"),
            "chunk_index": m.get("chunk_index"),
        }
        # if scheme filter applied, ensure match
        if where and chunk.get("scheme_key") != plan.scheme_key and plan.scheme_key is not None:
            # also allow cross-scheme? not specified; just count
            pass
        chunks.append(RetrievedChunk(chunk=chunk, score=float(score), rank=i+1))
    if len(chunks) > config.TOP_K:
        chunks = sorted(chunks, key=lambda x: x.score, reverse=True)[:config.TOP_K]
    confidence = chunks[0].score if chunks else 0.0
    return RetrievalResult(
        chunks=chunks,
        dropped_low_score=dropped_low,
        dropped_scheme_mismatch=dropped_scheme,
        confidence=confidence,
        used_filter=bool(where),
    )


def format_context(result: RetrievalResult) -> str:
    lines = []
    for rc in result.chunks:
        m = rc.chunk
        lines.append(f"[{rc.rank}] (source: {m.get('source_url')} | scheme: {m.get('scheme_name')} | section: {m.get('section')} | score: {rc.score:.2f})")
        lines.append(m.get('text', ''))
        lines.append('')
    return '\n'.join(lines)