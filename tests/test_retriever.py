"""Phase 4 - retrieval tests (IMPLEMENTATION.md section 4.3).

These search the real collection, so they skip when no fresh index is on disk
rather than passing vacuously on an empty result.
"""

from __future__ import annotations

import pytest

from src import config, query
from src.retriever import (
    EVAL_QUERIES,
    format_context,
    retrieve,
    similarity_floor_for,
)
from src.types import CROSS_SCHEME_KEY, RetrievedChunk, RetrievalResult
from src.vectorstore import IndexStaleError

#: The in-scope evaluation queries from PRD section 12 plus the fund-manager
#: case that arrived with the ``fund_manager`` intent. The last two rows are
#: refused by the Phase 5 guardrails before any embedding happens, so retrieval
#: is not the stage that decides them.
IN_SCOPE_QUERIES = tuple(
    case["q"] for case in EVAL_QUERIES if case["handled_by"] == "retrieval"
)

#: One query per scheme, used to prove the metadata filter never bleeds.
PER_SCHEME_QUERIES = {
    "hdfc-large-cap": "expense ratio of HDFC Large Cap Direct Growth",
    "hdfc-equity-flexi-cap": "exit load of HDFC Equity Flexi Cap Direct Growth",
    "hdfc-elss-tax-saver": "lock-in period of HDFC ELSS Tax Saver",
    "hdfc-small-cap": "minimum lump sum of HDFC Small Cap Direct Growth",
    "hdfc-balanced-advantage": "riskometer and benchmark of HDFC Balanced Advantage",
}


# --------------------------------------------------------------------------- #
# Core retrieval
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw_query", IN_SCOPE_QUERIES)
def test_each_in_scope_query_returns_a_chunk_above_the_floor(
    raw_query: str, index_ready: None
) -> None:
    plan = query.build_plan(raw_query)
    result = retrieve(plan)
    assert result.chunks, f"no chunk survived for {raw_query!r}"
    assert result.confidence >= similarity_floor_for(plan)


def test_top_k_is_respected(index_ready: None) -> None:
    plan = query.build_plan("expense ratio of HDFC Large Cap Direct Growth")
    result = retrieve(plan)
    assert len(result.chunks) <= config.TOP_K


def test_chunks_are_sorted_by_descending_score(index_ready: None) -> None:
    plan = query.build_plan("minimum SIP amount of HDFC ELSS Tax Saver")
    scores = [retrieved.score for retrieved in retrieve(plan).chunks]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_contiguous_and_one_based(index_ready: None) -> None:
    plan = query.build_plan("exit load of HDFC Equity Flexi Cap Direct Growth")
    ranks = [retrieved.rank for retrieved in retrieve(plan).chunks]
    assert ranks == list(range(1, len(ranks) + 1))


def test_scores_are_bounded_cosine_similarities(index_ready: None) -> None:
    plan = query.build_plan("benchmark of HDFC Balanced Advantage")
    for retrieved in retrieve(plan).chunks:
        assert -1.0 <= retrieved.score <= 1.0


# --------------------------------------------------------------------------- #
# Scheme filter (D4)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("scheme_key", "raw_query"), list(PER_SCHEME_QUERIES.items()))
def test_top_chunk_belongs_to_the_queried_scheme(
    scheme_key: str, raw_query: str, index_ready: None
) -> None:
    """A Small Cap exit-load answer must never bleed into a Large Cap question."""
    plan = query.build_plan(raw_query)
    assert plan.scheme_key == scheme_key
    result = retrieve(plan)
    assert result.chunks, f"no chunk for {raw_query!r}"
    assert result.chunks[0].scheme_key == scheme_key


@pytest.mark.parametrize(("scheme_key", "raw_query"), list(PER_SCHEME_QUERIES.items()))
def test_used_filter_is_set_when_the_scheme_resolves(
    scheme_key: str, raw_query: str, index_ready: None
) -> None:
    assert retrieve(query.build_plan(raw_query)).used_filter is True


def test_no_filter_when_no_scheme_resolves(index_ready: None) -> None:
    result = retrieve(query.build_plan("How do I download my capital gains statement?"))
    assert result.used_filter is False


def test_filter_keeps_cross_scheme_documents_reachable(index_ready: None) -> None:
    """The RTA statement procedure is scheme-agnostic but must stay findable."""
    result = retrieve(query.build_plan("How do I download my capital gains statement?"))
    assert any(
        retrieved.scheme_key == CROSS_SCHEME_KEY for retrieved in result.chunks
    )


def test_cross_scheme_document_is_reachable_under_a_scheme_filter(
    index_ready: None,
) -> None:
    plan = query.build_plan(
        "How do I download my capital gains statement for HDFC Small Cap?"
    )
    result = retrieve(plan)
    assert result.used_filter is True
    assert any(
        retrieved.scheme_key == CROSS_SCHEME_KEY for retrieved in result.chunks
    )


def test_no_scheme_mismatch_when_a_filter_was_applied(index_ready: None) -> None:
    for raw_query in PER_SCHEME_QUERIES.values():
        result = retrieve(query.build_plan(raw_query))
        assert result.dropped_scheme_mismatch == 0


# --------------------------------------------------------------------------- #
# Similarity floor
# --------------------------------------------------------------------------- #


def test_nonsense_query_returns_nothing(index_ready: None) -> None:
    """Better an honest refusal than a weak match (PRD D1)."""
    result = retrieve(query.build_plan("what is the atmospheric pressure on Jupiter"))
    assert result.chunks == []
    assert result.confidence == 0.0
    assert result.is_empty


def test_unknown_intent_uses_the_stricter_floor() -> None:
    unknown_plan = query.build_plan("what is the atmospheric pressure on Jupiter")
    known_plan = query.build_plan("what is the expense ratio of HDFC Large Cap")
    assert similarity_floor_for(unknown_plan) == config.UNKNOWN_INTENT_FLOOR
    assert similarity_floor_for(known_plan) == config.SIMILARITY_FLOOR
    assert config.UNKNOWN_INTENT_FLOOR > config.SIMILARITY_FLOOR


def test_dropped_low_score_is_counted(index_ready: None) -> None:
    result = retrieve(query.build_plan("what is the atmospheric pressure on Jupiter"))
    assert result.dropped_low_score > 0, "candidates were dropped but not counted"


def test_no_chunk_below_the_applied_floor(index_ready: None) -> None:
    plan = query.build_plan("benchmark and riskometer of HDFC Balanced Advantage")
    floor = similarity_floor_for(plan)
    for retrieved in retrieve(plan).chunks:
        assert retrieved.score >= floor


def test_confidence_equals_the_top_surviving_score(index_ready: None) -> None:
    plan = query.build_plan("exit load of HDFC Small Cap Direct Growth")
    result = retrieve(plan)
    assert result.confidence == result.chunks[0].score


# --------------------------------------------------------------------------- #
# Stale-index guard
# --------------------------------------------------------------------------- #


def test_retrieve_raises_on_a_missing_index(monkeypatch) -> None:
    from src import retriever

    monkeypatch.setattr(retriever, "read_index_meta", lambda: None)
    with pytest.raises(IndexStaleError):
        retriever.retrieve(query.build_plan("expense ratio"))


def test_retrieve_raises_on_a_fingerprint_mismatch(monkeypatch, index_ready: None) -> None:
    """Never silently compare vectors from two different models (ARCH 4.6)."""
    from src import retriever

    monkeypatch.setattr(
        retriever, "is_index_stale", lambda: (True, "fingerprint mismatch")
    )
    with pytest.raises(IndexStaleError):
        retriever.retrieve(query.build_plan("expense ratio"))


# --------------------------------------------------------------------------- #
# Chunk records and context formatting
# --------------------------------------------------------------------------- #


def test_retrieved_chunk_carries_the_full_record(index_ready: None) -> None:
    """Citation and the extractive fallback index into these fields by name."""
    result = retrieve(query.build_plan("exit load of HDFC Large Cap"))
    chunk = result.chunks[0].chunk
    for key in (
        "chunk_id",
        "text",
        "scheme_key",
        "scheme_name",
        "category",
        "section",
        "label",
        "source_url",
        "source_title",
        "fetched_at",
    ):
        assert chunk.get(key), f"retrieved chunk is missing {key!r}"


def test_retrieved_source_urls_are_allow_listed(index_ready: None) -> None:
    """The citation verifier is the last line of defence; this is the first."""
    for raw_query in IN_SCOPE_QUERIES:
        for retrieved in retrieve(query.build_plan(raw_query)).chunks:
            assert retrieved.source_url in config.ALLOWED_URLS


def test_format_context_uses_the_documented_block_format(index_ready: None) -> None:
    result = retrieve(query.build_plan("expense ratio of HDFC Large Cap"))
    context = format_context(result)
    first = context.split("\n", 1)[0]
    assert first.startswith("[1] (source: http")
    assert " | scheme: " in first
    assert " | section: " in first
    assert " | score: " in first
    assert ")" in first


def test_format_context_numbers_every_block(index_ready: None) -> None:
    context = format_context(retrieve(query.build_plan("exit load of HDFC Large Cap")))
    ranks = [
        line.split("]")[0]
        for line in context.splitlines()
        if line.startswith("[") and "] (source:" in line
    ]
    assert ranks == [f"[{n}" for n in range(1, len(ranks) + 1)]


def test_format_context_includes_the_chunk_text(index_ready: None) -> None:
    result = retrieve(query.build_plan("expense ratio of HDFC Large Cap"))
    context = format_context(result)
    assert result.chunks[0].text in context


def test_format_context_is_empty_for_an_empty_result() -> None:
    empty = RetrievalResult(
        chunks=[],
        dropped_low_score=3,
        dropped_scheme_mismatch=0,
        confidence=0.0,
        used_filter=False,
    )
    assert format_context(empty) == ""


def test_retrieval_result_reports_distinct_schemes(index_ready: None) -> None:
    result = retrieve(query.build_plan("How do I download my capital gains statement?"))
    assert CROSS_SCHEME_KEY in result.schemes or result.schemes


def test_result_is_empty_safe(index_ready: None) -> None:
    empty = RetrievalResult(
        chunks=[RetrievedChunk(chunk={"scheme_key": "x"}, score=0.9, rank=1)],
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=0.9,
        used_filter=False,
    )
    assert not empty.is_empty
    assert empty.schemes == ["x"]