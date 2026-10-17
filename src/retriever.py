"""Stages 6-8 - Retrieval (ARCHITECTURE.md section 4.6).

Two rules govern this file:

1. Never search an index built by a different embedding space. A fingerprint
   mismatch raises :class:`IndexStaleError` rather than returning plausible
   nonsense from a foreign vector space.
2. Keep the value with its label. The scheme metadata filter stops a Small Cap
   exit-load answer bleeding into a Large Cap question (D4), and the similarity
   floor is what lets guardrail B refuse with "I don't have that in my sources"
   instead of generating from an unrelated chunk.
"""

from __future__ import annotations

import re
import time

from chromadb.api.models.Collection import Collection

from src import config
from src.embedding import embed_query, embedding_fingerprint, get_embeddings
from src.types import CROSS_SCHEME_KEY, QueryPlan, RetrievedChunk, RetrievalResult
from src.vectorstore import (
    IndexStaleError,
    get_collection,
    is_index_stale,
    read_index_meta,
)

__all__ = ["retrieve", "format_context", "similarity_floor_for", "run_all", "main"]


def similarity_floor_for(plan: QueryPlan) -> float:
    """Return the floor that applies to a specific query.

    ``SIMILARITY_FLOOR`` is the default. A query whose intent resolved to
    ``"unknown"`` uses ``UNKNOWN_INTENT_FLOOR``, which is stricter: when we do
    not know what is being asked, a merely plausible-looking match is exactly
    the case we want to refuse (ARCH section 4.6, "raise the floor when intent
    is unknown").

    Delegates to :func:`src.config.similarity_floor_for` so that guardrail
    phase B applies the identical threshold without importing this module
    (IMPLEMENTATION.md ground rule 4).

    Args:
        plan: The rewritten query.

    Returns:
        The minimum cosine similarity a chunk must reach to survive.
    """
    return config.similarity_floor_for(plan.intent)


def _assert_fresh() -> None:
    """Fail loudly when the index and the query embedder disagree.

    Raises:
        IndexStaleError: When ``data/index_meta.json`` is missing or its
            fingerprint does not match :func:`embedding_fingerprint`.
    """
    stale, reason = is_index_stale()
    if stale:
        raise IndexStaleError(
            f"index is stale ({reason}); rebuild with "
            "`python run_index.py --stage index --rebuild`"
        )
    meta = read_index_meta() or {}
    if meta.get("embedding_fingerprint") != embedding_fingerprint():  # pragma: no cover
        raise IndexStaleError("embedding fingerprint mismatch")


def _scheme_filter(scheme_key: str) -> dict[str, str | list[dict[str, str]]]:
    """Build the Chroma metadata filter for a resolved scheme.

    Cross-scheme documents (the RTA capital-gains statement procedure,
    ``scheme_key == "all-schemes"``) stay visible under every scheme filter, so
    a scheme-scoped question about statements can still find them.

    Args:
        scheme_key: The canonical scheme key resolved by Stage 5.

    Returns:
        A ``where`` clause for ``collection.query``.
    """
    return {
        "$or": [
            {"scheme_key": scheme_key},
            {"scheme_key": CROSS_SCHEME_KEY},
        ]
    }


def retrieve(plan: QueryPlan) -> RetrievalResult:
    """Search the collection for chunks that answer a rewritten query.

    The pipeline is: freshness check, embed the expanded query, Chroma search
    with the scheme filter, convert cosine distance to similarity, sort, keep the
    top ``TOP_K``, then apply the similarity floor.

    Args:
        plan: The Stage 5 output.

    Returns:
        A :class:`~src.types.RetrievalResult`. ``chunks`` is empty when nothing
        cleared the floor, and ``confidence`` is then ``0.0`` - that empty
        result is the signal guardrail B turns into a low-confidence refusal.

    Raises:
        IndexStaleError: When the persisted index was built by a different
            embedding space.
    """
    _assert_fresh()
    where = _scheme_filter(plan.scheme_key) if plan.scheme_key else None
    collection = get_collection()
    query_vector = embed_query(plan.expanded)
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=config.RETRIEVE_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    candidates = _candidates(raw)
    candidates = _boost_label_matches(plan, candidates)
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[: config.TOP_K]

    dropped_scheme_mismatch = 0
    if plan.scheme_key:
        kept = []
        for score, chunk in candidates:
            # Defence in depth: Chroma already filtered on scheme_key, but a
            # cross-scheme document is legitimately in scope for any scheme.
            if chunk["scheme_key"] in (plan.scheme_key, CROSS_SCHEME_KEY):
                kept.append((score, chunk))
            else:
                dropped_scheme_mismatch += 1
        candidates = kept

    floor = similarity_floor_for(plan)
    dropped_low_score = 0
    surviving: list[RetrievedChunk] = []
    for score, chunk in candidates:
        if score < floor:
            dropped_low_score += 1
            continue
        surviving.append(
            RetrievedChunk(chunk=chunk, score=round(score, 4), rank=len(surviving) + 1)
        )

    confidence = surviving[0].score if surviving else 0.0
    return RetrievalResult(
        chunks=surviving,
        dropped_low_score=dropped_low_score,
        dropped_scheme_mismatch=dropped_scheme_mismatch,
        confidence=confidence,
        used_filter=where is not None,
    )


def _boost_label_matches(
    plan: QueryPlan, candidates: list[tuple[float, dict]]
) -> list[tuple[float, dict]]:
    """Add a relevance bonus to candidates whose label names the asked-for fact.

    Scheme-name questions ("exit load of HDFC Large Cap") embed the scheme title
    into every chunk, so a correctly-labelled fact chunk can score just under the
    floor while the generic "About" overview clears it. The bonus is applied to
    the same ``label`` + ``section`` surface that guardrail B grounds on, which
    keeps retrieval and grounding in agreement about what "carries the fact"
    means. Scores are clamped to ``1.0`` so the reported value stays bounded.

    Args:
        plan: The Stage 5 plan, whose intent selects the expected labels.
        candidates: ``(score, chunk)`` pairs straight from :func:`_candidates`.

    Returns:
        The same pairs, with a bonus added where a label matched.
    """
    labels = config.FACT_TYPES.get(plan.intent, {}).get("expected_labels", [])
    if not labels:
        return candidates
    boosted: list[tuple[float, dict]] = []
    for score, chunk in candidates:
        surface = f"{chunk.get('label', '')} {chunk.get('section', '')}".lower()
        if any(_matches_label(surface, label) for label in labels):
            score = min(1.0, score + config.RETRIEVAL_LABEL_BOOST)
        boosted.append((score, chunk))
    return boosted


def _matches_label(surface: str, label: str) -> bool:
    """True when ``label`` appears in ``surface`` as a whole word or phrase.

    Word boundaries matter: without them the exit-load label ``load`` would match
    the "How to Download a Capital Gains Statement" section of the cross-scheme
    document and boost it to the top of every exit-load query.
    """
    if not label:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in label.split()) + r"\b"
    return re.search(pattern, surface) is not None


def _candidates(raw: dict) -> list[tuple[float, dict]]:
    """Unpack a Chroma response into ``(similarity, chunk_record)`` pairs.

    ``chunk_id`` and ``text`` come from the query response and everything else
    from the stored metadata, so a retrieved chunk is the same shape as a record
    in ``chunks.jsonl``. Downstream stages (citation, extractive fallback) index
    into it by name.

    Args:
        raw: The dict returned by ``collection.query``.

    Returns:
        Pairs in the order Chroma returned them; the caller sorts.
    """
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    pairs: list[tuple[float, dict]] = []
    for position, distance in enumerate(distances):
        if position >= len(metadatas):
            break
        metadata = dict(metadatas[position] or {})
        chunk = dict(metadata)
        chunk.setdefault("chunk_id", ids[position] if position < len(ids) else "")
        chunk.setdefault(
            "text", documents[position] if position < len(documents) else ""
        )
        pairs.append((1.0 - float(distance), chunk))
    return pairs


def format_context(result: RetrievalResult) -> str:
    """Render retrieved chunks as the numbered context blocks the prompt expects.

    The format is fixed by ARCH section 4.6 step 6 so the generator sees a
    stable, numbered citation space:

    ``[1] (source: <url> | scheme: <name> | section: <s> | score: 0.71)``
    followed by the chunk text.

    Args:
        result: The retrieval outcome.

    Returns:
        The context string, or an empty string when nothing survived.
    """
    blocks: list[str] = []
    for retrieved in result.chunks:
        chunk = retrieved.chunk
        header = (
            f"[{retrieved.rank}] (source: {chunk.get('source_url', '')}"
            f" | scheme: {chunk.get('scheme_name', '')}"
            f" | section: {chunk.get('section', '')}"
            f" | score: {retrieved.score:.2f})"
        )
        blocks.append(f"{header}\n{chunk.get('text', '')}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# CLI: retrieval quality report
# --------------------------------------------------------------------------- #

#: The PRD section 12 evaluation set, used as the Phase 4 threshold-tuning
#: table. ``handled_by`` records which stage decides the case:
#: ``"retrieval"`` means this stage must return chunks, ``"guardrails_a"``
#: means the query is refused before any embedding happens (Phase 5), so its
#: chunk count is reported for information only and is not scored here.
EVAL_QUERIES: tuple[dict, ...] = (
    {"q": "expense ratio of HDFC Large Cap Direct Growth", "intent": "expense_ratio", "handled_by": "retrieval"},
    {"q": "exit load of HDFC Equity Flexi Cap Direct Growth", "intent": "exit_load", "handled_by": "retrieval"},
    {"q": "minimum SIP amount of HDFC ELSS Tax Saver Direct Growth", "intent": "min_sip", "handled_by": "retrieval"},
    {"q": "lock-in period of HDFC ELSS Tax Saver", "intent": "lock_in", "handled_by": "retrieval"},
    {"q": "riskometer and benchmark of HDFC Balanced Advantage", "intent": "riskometer", "handled_by": "retrieval"},
    {"q": "minimum lump sum of HDFC Small Cap Direct Growth", "intent": "min_lump_sum", "handled_by": "retrieval"},
    {"q": "How do I download my capital gains statement?", "intent": "statement_download", "handled_by": "retrieval"},
    {"q": "What is the current NAV of HDFC Large Cap?", "intent": "nav", "handled_by": "retrieval"},
    {"q": "When was HDFC Large Cap Fund launched?", "intent": "inception", "handled_by": "retrieval"},
    {"q": "Who is the fund manager of HDFC Large Cap?", "intent": "fund_manager", "handled_by": "retrieval"},
    {"q": "Should I buy HDFC Small Cap?", "intent": "unknown", "handled_by": "guardrails_a"},
    {"q": "what is the atmospheric pressure on Jupiter", "intent": "unknown", "handled_by": "guardrails_a"},
)


def _eval_report() -> list[dict]:
    """Run the evaluation set and return one row per query.

    The embedder is warmed up first: the very first ``embed_query`` call pays a
    one-time model load (tens of seconds from a cold process), which says
    nothing about retrieval speed. In the app that cost is paid once per process
    behind ``@st.cache_resource``.

    Returns:
        Rows with the resolved intent, chunk count, top similarity and latency.
    """
    from src import query

    get_embeddings()  # warm up: exclude the one-time model load from the timings
    embed_query("warmup")

    rows: list[dict] = []
    for case in EVAL_QUERIES:
        plan = query.build_plan(case["q"])
        started = time.perf_counter()
        result = retrieve(plan)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "q": case["q"],
                "intent": plan.intent,
                "scheme": plan.scheme_key,
                "handled_by": case["handled_by"],
                "chunks": len(result.chunks),
                "top1": result.confidence,
                "floor": similarity_floor_for(plan),
                "filtered": result.used_filter,
                "dropped_low": result.dropped_low_score,
                "label": result.chunks[0].label if result.chunks else "",
            }
        )
    # Time a repeat pass so the latency figure reflects steady state.
    timed = _latency_pass()
    rows[0]["ms"] = timed
    return rows


def _latency_pass(repeats: int = 5) -> float:
    """Return the mean retrieval-only latency in milliseconds.

    Args:
        repeats: How many in-scope queries to time.

    Returns:
        Mean wall-clock milliseconds per ``retrieve`` call, embedder excluded.
    """
    from src import query

    samples: list[float] = []
    for case in EVAL_QUERIES[:repeats]:
        plan = query.build_plan(case["q"])
        started = time.perf_counter()
        retrieve(plan)
        samples.append((time.perf_counter() - started) * 1000)
    return sum(samples) / len(samples) if samples else 0.0


def _print_report(rows: list[dict]) -> bool:
    """Print the threshold-tuning table and return whether it passes."""
    print(
        f"{'query':<52} {'intent':<18} {'top1':>6} {'floor':>6} "
        f"{'chunks':>6} {'handled by':<12}"
    )
    print("-" * 108)
    ok = True
    for row in rows:
        print(
            f"{row['q'][:50]:<52} {row['intent']:<18} "
            f"{row['top1']:>6.3f} {row['floor']:>6.2f} {row['chunks']:>6d} "
            f"{row['handled_by']:<12}"
        )
        if row["handled_by"] == "retrieval" and row["chunks"] == 0:
            ok = False
    print("-" * 108)
    in_scope = [row for row in rows if row["handled_by"] == "retrieval"]
    if in_scope:
        scores = [row["top1"] for row in in_scope]
        print(
            f"in-scope top-1 similarity: min {min(scores):.3f}, "
            f"max {max(scores):.3f} over {len(scores)} queries"
        )
    refused = [row for row in rows if row["handled_by"] == "guardrails_a"]
    for row in refused:
        print(
            f"pre-retrieval refusal ({row['q'][:40]!r}): "
            f"top-1 {row['top1']:.3f} - not scored, decided in Phase 5"
        )
    latency_ms = rows[0].get("ms", 0.0)
    print(f"mean retrieval-only latency: {latency_ms:.0f} ms (NFR2 budget 1500 ms)")
    print(
        f"SIMILARITY_FLOOR = {config.SIMILARITY_FLOOR}, "
        f"UNKNOWN_INTENT_FLOOR = {config.UNKNOWN_INTENT_FLOOR}"
    )
    print("result:", "PASS" if ok and latency_ms < 1500 else "FAIL")
    return ok and latency_ms < 1500


def run_all() -> int:
    """``run_index.py --stage eval-retrieval`` entry point.

    Returns:
        ``0`` when every expectation held and latency fit NFR2, else ``1``.
    """
    stale, reason = is_index_stale()
    if stale:
        print(f"  index is stale ({reason}) - run --stage index --rebuild first")
        return 1
    rows = _eval_report()
    return 0 if _print_report(rows) else 1


def main() -> int:
    """Alias of :func:`run_all` so the stage dispatcher can call either."""
    return run_all()