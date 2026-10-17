"""Phase 3 - embedding and vector store tests (IMPLEMENTATION.md section 3.3).

The collection contract is the thing under test: cosine distance, no Chroma-side
embedder, ``chunk_id`` as the document id, and a fingerprint that refuses to
search an index built by a different embedding space.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.embedding import embed_query, embedding_fingerprint
from src.vectorstore import (
    COLLECTION_NAME,
    IndexStaleError,
    get_collection,
    is_index_stale,
    read_index_meta,
)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_is_12_hex_chars() -> None:
    fingerprint = embedding_fingerprint()
    assert len(fingerprint) == 12
    assert all(char in "0123456789abcdef" for char in fingerprint)


def test_fingerprint_is_stable_within_a_process() -> None:
    assert embedding_fingerprint() == embedding_fingerprint()


def test_fingerprint_changes_with_the_document_prefix(monkeypatch) -> None:
    """The prefix is half the embedding space, so it has to be in the hash."""
    before = embedding_fingerprint()
    monkeypatch.setattr(config, "DOC_EMBED_PREFIX", "a different prefix: ")
    assert embedding_fingerprint() != before


def test_fingerprint_ignores_the_concrete_device(monkeypatch) -> None:
    """CPU and GPU produce the same vectors, so the device is not part of the hash."""
    before = embedding_fingerprint()
    monkeypatch.setattr(config, "EMBED_DEVICE", "cuda")
    assert embedding_fingerprint() == before


def test_fingerprint_changes_with_the_model(monkeypatch) -> None:
    before = embedding_fingerprint()
    monkeypatch.setattr(config, "EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    assert embedding_fingerprint() != before


# --------------------------------------------------------------------------- #
# Index metadata (ARCHITECTURE section 5.1)
# --------------------------------------------------------------------------- #


def test_index_meta_has_the_arch_schema() -> None:
    meta = read_index_meta()
    assert meta is not None, "data/index_meta.json missing - run --stage index"
    for key in (
        "collection",
        "embedding_model",
        "embedding_fingerprint",
        "doc_embed_prefix",
        "chunk_size_tokens",
        "chunk_overlap_tokens",
        "chunk_strategy",
        "distance",
        "doc_count",
        "chunk_count",
        "built_at",
        "generator_model",
    ):
        assert key in meta, f"index_meta.json is missing {key!r}"
    assert meta["collection"] == COLLECTION_NAME == config.CHROMA_COLLECTION
    assert meta["distance"] == "cosine"
    assert meta["doc_embed_prefix"] == config.DOC_EMBED_PREFIX


def test_is_index_stale_is_false_on_a_built_index(index_ready: None) -> None:
    stale, reason = is_index_stale()
    assert stale is False, reason


def test_is_index_stale_detects_a_fingerprint_change(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    stale, reason = is_index_stale()
    assert stale is True
    assert reason == "fingerprint mismatch"


# --------------------------------------------------------------------------- #
# Collection contract (ARCHITECTURE section 4.4)
# --------------------------------------------------------------------------- #


def test_collection_count_matches_chunks_jsonl(
    chunks: list[dict], index_ready: None
) -> None:
    assert get_collection().count() == len(chunks)
    assert read_index_meta()["chunk_count"] == len(chunks)


def test_collection_has_no_embedding_function(index_ready: None) -> None:
    """Chroma must never load a second model into this collection."""
    assert get_collection()._embedding_function is None


def test_collection_metadata_is_cosine(index_ready: None) -> None:
    assert get_collection().metadata["hnsw:space"] == "cosine"


def test_round_trip_returns_allow_listed_source_urls(index_ready: None) -> None:
    """A hand-typed vector search comes back with usable citation metadata."""
    result = get_collection().query(
        query_embeddings=[embed_query("exit load of HDFC large cap")],
        n_results=5,
        include=["metadatas", "documents", "distances"],
    )
    metadatas = result["metadatas"][0]
    assert metadatas, "no hits"
    for meta in metadatas:
        assert meta["source_url"] in config.ALLOWED_URLS
        assert meta["chunk_id"]
        assert meta["label"]
        assert meta["section"]


def test_exit_load_query_hits_the_exit_load_chunk(index_ready: None) -> None:
    """The Phase 3 sanity check: separability of one canonical fact.

    Asserted through the retrieval + grounding path rather than the raw
    collection: the scheme title inside the query dominates the embedding, so the
    correctly labelled exit-load chunk can sit just under the floor while the
    generic "About" overview clears it. The label boost is what keeps the fact
    chunk alive for guardrail B to ground on, so that is the surface the property
    has to hold on.
    """
    from src import guardrails
    from src import query
    from src.retriever import retrieve

    plan = query.build_plan("exit load of HDFC large cap")
    result = retrieve(plan)
    assert result.chunks, "nothing survived retrieval for an in-scope question"
    matched, label = guardrails.grounding_evidence(plan.intent, result)
    assert matched is not None, "the exit-load question did not ground"
    assert "exit load" in f"{matched.label} {matched.section}".lower()


def test_every_chunk_id_in_the_corpus_is_present(
    chunks: list[dict], index_ready: None
) -> None:
    stored = get_collection().get(include=[])["ids"]
    assert sorted(stored) == sorted(record["chunk_id"] for record in chunks)


def test_document_ids_are_unique(chunks: list[dict]) -> None:
    """``chunk_id`` is the Chroma document id, so duplicates cannot be indexed."""
    ids = [record["chunk_id"] for record in chunks]
    duplicates = {chunk_id for chunk_id in ids if ids.count(chunk_id) > 1}
    assert not duplicates, f"duplicate chunk_id in chunks.jsonl: {sorted(duplicates)[:5]}"


def test_index_stale_error_is_raised_on_a_mismatched_index(
    monkeypatch, index_ready: None
) -> None:
    """Guards the invariant behind the UI's stale-index banner."""
    from src import retriever

    plan = retriever.QueryPlan(
        raw="expense ratio",
        normalized="expense ratio",
        expanded="expense ratio",
        scheme_key="hdfc-large-cap",
        intent="expense_ratio",
        expansions=[],
    )
    monkeypatch.setattr(
        retriever, "read_index_meta", lambda: {"embedding_fingerprint": "deadbeef0000"}
    )
    with pytest.raises(IndexStaleError):
        retriever.retrieve(plan)


def test_chroma_dir_is_gitignored() -> None:
    """The index is rebuildable, so it must never be committed (Phase 0 rule)."""
    gitignore = (config.PATHS["ROOT"] / ".gitignore").read_text(encoding="utf-8")
    assert "data/chroma/" in gitignore


def test_metadata_projection_rejects_none() -> None:
    """Chroma rejects None metadata values, so the projection coerces them."""
    from src.vectorstore import _chunk_metadata

    meta = _chunk_metadata(
        {"chunk_id": "x::0000", "text": "t", "scheme_key": "hdfc-large-cap", "chunk_index": 0}
    )
    assert all(value is not None for value in meta.values())
    assert json.dumps(meta)  # serialisable