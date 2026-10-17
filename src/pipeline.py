"""Stages 5-11 in one call, plus the cache, the log and the smoke test.

This is the only module that knows the order of the stages. Everything it calls is a
pure function that can be tested alone: the plan (stage 5), the guardrails, the
retriever, the generator, the citation finaliser. The pipeline's own job is the
sequencing, the timings, the trace and the two things that have to happen no matter
how far it gets - logging every query, and never returning a message that breaks a
rendering invariant.

**Why the extractive path exists.** The answer path is allowed to be a plain extract
when no key is configured, the key is missing, the model errors, or the model
disobeys the JSON contract (``EXTRACTIVE_ONLY``). A 429 from a free-tier key must not
turn a working demo into a stack trace. It is a *degradation*, not a separate
product: the extractive body is built only out of verbatim sentences from the chosen
chunk, and it goes through the identical sentence cap, URL stripping, link resolution
and invariant check as an LLM answer.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from src import citation, config, guardrails, logger, query
from src.generator import build_prompt, generate, llm_available
from src.retriever import format_context, retrieve, similarity_floor_for
from src.types import (
    AssistantMessage,
    LLMResponse,
    PipelineResult,
    QueryPlan,
    RefusalResult,
    RetrievedChunk,
    RetrievalResult,
)

__all__ = [
    "answer",
    "answer_extractive",
    "extract_sentences",
    "clear_cache",
    "run_all",
    "main",
]

#: marker shown in the trace when the answer came from the corpus, not the model
EXTRACTIVE_MODE = "extractive"

#: Written into :attr:`src.types.LLMResponse.raw` for extractive answers so the UI
#: trace makes it obvious that no model was called.
EXTRACTIVE_RAW = "[extractive: no model call]"

#: sentence key ``cache`` -> (result, expiry, retrieval outcome). Process-local by
#: design: it is a demo cache, and a shared cache would need the PII redaction to be
#: airtight at write time. The retrieval outcome travels with the entry so a cache hit
#: can log the same ``sources`` and ``top_score`` the original query logged.
_CACHE: dict[str, tuple[PipelineResult, float, RetrievalResult | None]] = {}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def clear_cache() -> int:
    """Drop every cached answer and return how many were dropped."""
    count = len(_CACHE)
    _CACHE.clear()
    return count


def _cache_get(key: str) -> PipelineResult | None:
    """Return a fresh cached answer for ``key``, evicting it if it expired."""
    entry = _CACHE.get(key)
    if entry is None:
        return None
    result, expires_at, _retrieval = entry
    if expires_at <= time.time():
        del _CACHE[key]
        return None

    # Re-serve the same immutable content, but marked as a cache hit so the UI and
    # the log can prove the fast path exists.
    cached = AssistantMessage(
        kind=result.message.kind,
        text=result.message.text,
        source_url=result.message.source_url,
        last_updated=result.message.last_updated,
        disclaimer=result.message.disclaimer,
        truncated=result.message.truncated,
        link_substituted=result.message.link_substituted,
        cached=True,
    )
    citation.assert_invariants(cached)
    return PipelineResult(cached, dict(result.trace))


def _cache_put(
    key: str, result: PipelineResult, retrieval: RetrievalResult | None
) -> None:
    """Cache ``result`` under the expanded query for the configured TTL."""
    _CACHE[key] = (result, time.time() + config.CACHE_TTL_SECONDS, retrieval)


# --------------------------------------------------------------------------- #
# Extractive path
# --------------------------------------------------------------------------- #


def _clean_text(text: str) -> str:
    """Drop headings and metadata lines, then collapse whitespace.

    Chunk bodies are extracted from a document that keeps its own layout: a ``##``
    heading on its own line, ``> category: ...`` metadata, hard-wrapped paragraphs,
    and a stray ``;`` where the source table ended. Left in, a heading is glued onto
    the front of the first sentence ("### About HDFC Small Cap Fund About HDFC Small
    Cap Fund is a ...") and the answer reads as a bug.

    Args:
        text: Raw chunk text.

    Returns:
        One whitespace-normalised line with the layout noise removed.
    """
    kept = [
        line
        for line in text.splitlines()
        if line.strip() and not re.match(r"^\s*(#{1,6}\s|>|\||;)", line)
    ]
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def _terminal(text: str) -> str:
    """Ensure ``text`` ends with sentence punctuation so the cap counts it once."""
    text = text.strip().rstrip(";:,").strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        text = f"{text}."
    return text


def extract_sentences(text: str) -> list[str]:
    """Split a chunk body into clean, punctuated sentences.

    Args:
        text: Raw chunk text, layout and all.

    Returns:
        The sentences in document order, cleaned of markup and re-punctuated.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [_terminal(part) for part in parts if part and part not in {".", "!", "?"}]


def _is_label_line(line: str, labels: list[str]) -> bool:
    """Is this line one of the labels we are looking for, on its own?

    Equality, not containment. "Exit load of 1% if redeemed within 1 year" mentions
    the label but is a sentence, and treating it as a label is how an extractive
    answer ends up quoting a whole paragraph followed by " : ;".
    """
    candidate = line.strip().rstrip(":").strip().lower().replace("-", " ")
    if not candidate or len(candidate) > 40:
        return False
    return any(candidate == label.lower().replace("-", " ") for label in labels)


def _looks_like_value(line: str, labels: list[str]) -> bool:
    """Is this line the *value* that belongs to the label line above it?

    Short, has a real character in it, no sentence punctuation and no second
    ``label: value`` pair - so the stacked card's "NAV: 01 Oct '26" row is never
    mistaken for the value of the "Very High Risk" row above it.
    """
    candidate = line.strip()
    return bool(
        candidate
        and len(candidate) <= 60
        and re.search(r"[A-Za-z0-9]", candidate)
        and not candidate.endswith(".")
        and ":" not in candidate
        and not _is_label_line(candidate, labels)
    )


def _label_facts(text: str, labels: list[str]) -> list[str]:
    """Recover ``Label: value`` pairs whose label is one of ``labels``.

    The stacked scheme cards are laid out one field per line ("Expense ratio" on one
    line, "1.04%" on the next), so this is where the precise numbers actually live.
    The pair is rejoined into a sentence that still appears in the chunk once
    whitespace is normalised, so nothing is invented.

    Args:
        text: Raw chunk text.
        labels: Expected labels for the intent, in priority order.

    Returns:
        At most :data:`src.config.MAX_ANSWER_SENTENCES` facts.
    """
    if not labels:
        return []
    lines = [line.strip() for line in text.splitlines()]
    facts: list[str] = []
    for index, line in enumerate(lines[:-1]):
        if not _is_label_line(line, labels) or not _looks_like_value(lines[index + 1], labels):
            continue
        facts.append(f"{line.rstrip(':').strip()}: {lines[index + 1]}")
        if len(facts) == config.MAX_ANSWER_SENTENCES:
            break
    return facts


def _mentions(text: str, label: str) -> bool:
    """Case-insensitive containment that tolerates the ``exit load``/``exit-load``
    hyphenation and the ``top 100``/``top-100`` split in the corpus."""
    normalised = text.lower().replace("-", " ")
    return label.lower().replace("-", " ") in normalised


def _heading_fact(chunk: RetrievedChunk, labels: list[str]) -> str:
    """A ``Label: <heading>`` fact for a chunk whose heading carries the value.

    Some facts live only in the heading: a fund-manager block is a person's name
    and tenure ("DM Dhruv Muchhal Jun 2023 - Present View details") under a label
    that exists nowhere in the body. :func:`_clean_text` drops heading lines, so
    without this the extractive answer would quote the bio's experience sentence
    and never say who the manager is.

    Args:
        chunk: The chunk the extractive body is built from.
        labels: The intent's expected labels, for the grounding check.

    Returns:
        The fact, or ``""`` when the chunk's label does not ground on an expected
        label or the chunk has no heading line.
    """
    surface = f"{chunk.label} {chunk.section}"
    if not any(_mentions(surface, label) for label in labels):
        return ""
    match = re.search(r"^#{1,6}\s+(.+?)\s*$", chunk.text, re.MULTILINE)
    if not match:
        return ""
    heading = re.sub(r"\s+View details\s*$", "", match.group(1).strip())
    if not heading or _mentions(heading, chunk.label):
        return ""
    return f"{chunk.label}: {heading}"


def _query_terms(plan: QueryPlan) -> list[str]:
    """Content words of the query, for the fallback chunk pick."""
    stop = {
        "what", "is", "the", "of", "for", "in", "a", "an", "and", "or", "to", "my",
        "can", "do", "does", "how", "tell", "about", "which", "me", "any", "on",
        "fund", "funds", "mutual", "scheme", "growth", "direct", "plan",
    }
    words = re.findall(r"[a-z0-9%]+", plan.normalized.lower())
    return [word for word in words if word not in stop and len(word) > 2]


def _has_body(plan: QueryPlan, chunk: RetrievedChunk) -> bool:
    """Whether ``chunk`` yields a quotable extractive body.

    A document's title/metadata preamble becomes its own chunk (the ``#`` heading
    plus the ``> category: ...`` header lines). Once :func:`_clean_text` strips that
    layout noise there is nothing left to quote, so such a chunk must not win the
    pick merely because its label matches the intent - that is how "How do I
    download my capital gains statement?" produced a blank answer.
    """
    return bool(_extractive_body(plan, chunk).strip())


def _pick_chunk(plan: QueryPlan, result: RetrievalResult) -> RetrievedChunk:
    """Choose the chunk the extractive answer is built from.

    Order, most precise first:

    1. The chunk whose label/section matches an expected intent label *and* has a
       quotable body.
    2. Any other chunk that matches an expected label and has a body.
    3. The chunk with the most query terms in its body, if it has a body.
    4. The first chunk with a body.
    5. The top-ranked chunk, body or not (the empty-answer guard of last resort).

    Args:
        plan: The Stage 5 plan.
        result: The retrieval outcome. Non-empty by the time this is called, because
            phase B refuses an empty result first.

    Returns:
        The chunk to summarise. ``_has_body`` is what keeps a matched-but-empty
        preamble chunk from shadowing the content chunk behind it.
    """
    matched, _label = guardrails.grounding_evidence(plan.intent, result)
    if matched is not None and _has_body(plan, matched):
        return matched

    for label in guardrails.expected_labels(plan.intent):
        for retrieved in result.chunks:
            if retrieved is matched:
                continue
            if _mentions(
                f"{retrieved.label} {retrieved.section}", label
            ) and _has_body(plan, retrieved):
                return retrieved

    terms = _query_terms(plan)
    if terms:
        best = max(
            result.chunks,
            key=lambda retrieved: sum(
                1 for term in terms if _mentions(retrieved.text, term)
            ),
        )
        if any(_mentions(best.text, term) for term in terms) and _has_body(plan, best):
            return best

    for retrieved in result.chunks:
        if _has_body(plan, retrieved):
            return retrieved
    return result.chunks[0]


def _extractive_body(plan: QueryPlan, chunk: RetrievedChunk) -> str:
    """Build the answer body from ``chunk`` alone.

    Three passes, most precise first:

    1. ``Label: value`` pairs for the intent's expected labels.
    2. Sentences that mention an expected label.
    3. Sentences that mention a query term, then simply the first sentences.

    Every pass yields verbatim chunk content, so the "every number is in the context"
    property that the answer path relies on also holds here.

    Args:
        plan: The Stage 5 plan.
        chunk: The chunk to summarise.

    Returns:
        A body of at most :data:`src.config.MAX_ANSWER_SENTENCES` sentences.
    """
    limit = config.MAX_ANSWER_SENTENCES
    labels = guardrails.expected_labels(plan.intent)

    facts = _label_facts(chunk.text, labels)
    if facts:
        return " ".join(facts[:limit])

    sentences = extract_sentences(chunk.text)
    if labels:
        relevant = [
            sentence for sentence in sentences if any(_mentions(sentence, l) for l in labels)
        ]
        if relevant:
            return " ".join(relevant[:limit])
        heading_fact = _heading_fact(chunk, labels)
        if heading_fact:
            return heading_fact

    terms = _query_terms(plan)
    if terms:
        relevant = [
            sentence for sentence in sentences if any(_mentions(sentence, t) for t in terms)
        ]
        if relevant:
            return " ".join(relevant[:limit])

    return " ".join(sentences[:limit])


def answer_extractive(plan: QueryPlan, result: RetrievalResult) -> AssistantMessage:
    """Answer from the retrieved chunk without calling a model.

    Args:
        plan: The Stage 5 plan.
        result: The retrieval outcome.

    Returns:
        The finalised message, indistinguishable in shape from an LLM answer: same
        sentence cap, same verified link, same freshness stamp.

    Raises:
        guardrails.InvariantViolation: if the built body somehow violates a rendering
            invariant, which is a bug rather than bad input.
    """
    if result.is_empty:
        refusal = guardrails.render_refusal(
            "low_confidence", guardrails.educational_link("low_confidence", plan), "confidence.low"
        )
        return citation.finalise_refusal(refusal)

    chunk = _pick_chunk(plan, result)
    body = _extractive_body(plan, chunk)
    response = LLMResponse(
        answer=body,
        source_index=chunk.rank,
        insufficient_context=False,
        raw=EXTRACTIVE_RAW,
    )
    return citation.finalise(response, result)


# --------------------------------------------------------------------------- #
# Stage 9/10 helpers
# --------------------------------------------------------------------------- #


def _refuse(
    refusal: RefusalResult, plan: QueryPlan, trace: dict[str, Any], result: RetrievalResult | None = None
) -> PipelineResult:
    """Finalise a refusal, complete the trace and log it."""
    message = citation.finalise_refusal(refusal)
    trace["stage_11_citation"] = {
        "candidate_url": refusal.link,
        "verified": bool(refusal.link),
        "link_substituted": False,
        "last_updated": "",
        "invariants_ok": True,
    }
    citation.assert_invariants(message)
    logger.log_query(plan, result, _guardrail_trace(plan, trace), message)
    return PipelineResult(message, trace)


def _guardrail_trace(plan: QueryPlan, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten both guardrail stages into the list the logger reads."""
    entries = list(trace.get("guardrails_a") or []) + list(trace.get("guardrails_b") or [])
    return [entry for entry in entries if isinstance(entry, dict)]


def _new_trace(plan: QueryPlan, raw: str) -> dict[str, Any]:
    """Build the stage-5 part of the trace, with the raw query redacted.

    The trace is rendered in the UI, so it gets the same PII treatment as the log: a
    query that named an account number must not reappear in the "Show pipeline"
    expander either.

    ``guardrails_b`` starts empty rather than missing: a Phase A refusal never reached
    it, and a UI that renders the six stage keys blindly should see an empty list, not
    a ``KeyError``. ``stage_6_8_retrieval``, ``stage_10_generation`` and
    ``stage_11_citation`` are added by the stages that actually ran.
    """
    return {
        "stage_5_rewrite": {
            "raw": logger.redact(raw),
            "normalized": logger.redact(plan.normalized),
            "expanded": logger.redact(plan.expanded),
            "intent": plan.intent,
            "scheme_key": plan.scheme_key,
            "expansions": list(plan.expansions),
        },
        "guardrails_a": guardrails.describe_phase_a(raw),
        "guardrails_b": [],
    }


def _retrieval_trace(result: RetrievalResult, plan: QueryPlan) -> dict[str, Any]:
    """Stage 6-8 trace: what was fetched, what was dropped, what survived."""
    return {
        "floor": similarity_floor_for(plan),
        "top_k": config.TOP_K,
        "used_filter": result.used_filter,
        "dropped_low_score": result.dropped_low_score,
        "dropped_scheme_mismatch": result.dropped_scheme_mismatch,
        "top_score": result.confidence,
        "chunks": [
            {
                "rank": retrieved.rank,
                "score": round(retrieved.score, 4),
                "label": retrieved.chunk.get("label", ""),
                "section": retrieved.chunk.get("section", ""),
                "source_url": retrieved.source_url,
                "preview": str(retrieved.chunk.get("text", ""))[: config.CHUNK_PREVIEW_CHARS],
            }
            for retrieved in result.chunks
        ],
    }


# --------------------------------------------------------------------------- #
# answer
# --------------------------------------------------------------------------- #


def answer(query_text: str) -> PipelineResult:
    """Run one query through stages 5-11.

    Args:
        query_text: The raw user query.

    Returns:
        The message plus a complete trace. Every path ends with a logged line and an
        invariant check, including the ones that return early.
    """
    started = time.perf_counter()
    plan = query.build_plan(query_text)
    trace = _new_trace(plan, query_text)

    # --- Stage 9A: guardrails before anything else ----------------------- #
    # Deliberately first: a PAN should never be embedded, logged or cached.
    refusal_a = guardrails.check_phase_a(query_text, plan)
    if refusal_a is not None:
        return _refuse(refusal_a, plan, trace)

    # --- Cache ----------------------------------------------------------- #
    # Keyed on the *expanded* query so "nav" and "NAV of the fund" share an entry,
    # and looked up only after phase A so a refused query is never served from it.
    cache_key = plan.expanded.strip().lower()
    cached = _cache_get(cache_key)
    if cached is not None:
        cached.trace["cached"] = True
        total_ms = int((time.perf_counter() - started) * 1000)
        cached.trace["timings"] = {
            "retrieval_ms": 0,
            "llm_ms": 0,
            "total_ms": total_ms,
        }
        # A cache hit is still a query: it gets its own log line, with the sources and
        # score the original retrieval produced.
        logger.log_query(
            plan,
            _CACHE[cache_key][2],
            _guardrail_trace(plan, cached.trace),
            cached.message,
            {"total_ms": total_ms},
        )
        return cached

    # --- Stages 6-8: retrieval ------------------------------------------- #
    retrieval_started = time.perf_counter()
    result = retrieve(plan)
    retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)
    trace["stage_6_8_retrieval"] = _retrieval_trace(result, plan)

    # --- Stage 9B: confidence, NAV and grounding -------------------------- #
    trace["guardrails_b"] = guardrails.describe_phase_b(plan, result)
    refusal_b = guardrails.check_phase_b(plan, result)
    if refusal_b is not None:
        return _refuse(refusal_b, plan, trace, result)

    # --- Stage 10: generation, degrading instead of failing --------------- #
    # Render the exact prompt once, for the UI trace. It is recorded on both the
    # LLM and the extractive path so "Show pipeline" can prove grounding even when
    # no model was called (EXTRACTIVE_ONLY / offline demo). The context is public
    # corpus text and the query has already cleared Phase A, so nothing here is
    # personal data.
    prompt_text = build_prompt(plan, format_context(result))
    llm_started = time.perf_counter()
    mode = EXTRACTIVE_MODE
    reason = ""
    response: LLMResponse | None = None
    if llm_available():
        try:
            response = generate(plan, result)
            mode = "llm"
        except Exception as exc:  # noqa: BLE001 - see the docstring
            # Deliberately broad. This is the last line before the UI, the model
            # client is not ours, and the trace records the exception type and
            # message in ``reason`` - so a bug is still visible, but a free-tier
            # rate limit can no longer take the demo down with it.
            response = None
            reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = (
            "EXTRACTIVE_ONLY is set"
            if config.EXTRACTIVE_ONLY
            else "no GROQ_API_KEY in the environment"
        )
    llm_ms = int((time.perf_counter() - llm_started) * 1000)

    if mode == "llm" and response is not None and response.insufficient_context:
        refusal = guardrails.render_refusal(
            "low_confidence",
            guardrails.educational_link("low_confidence", plan),
            "confidence.low",
        )
        trace["stage_10_generation"] = {
            "mode": "llm",
            "model": config.GENERATOR_MODEL,
            "temperature": config.GENERATOR_TEMPERATURE,
            "prompt": prompt_text,
            "raw_response": response.raw,
            "insufficient_context": True,
            "source_index": response.source_index,
            "truncated": False,
        }
        return _refuse(refusal, plan, trace, result)

    if mode == "llm" and response is not None:
        message = citation.finalise(response, result)
    else:
        message = answer_extractive(plan, result)
    truncated = message.truncated
    trace["stage_10_generation"] = {
        "mode": mode,
        "reason": reason,
        "model": config.GENERATOR_MODEL if mode == "llm" else "",
        "temperature": config.GENERATOR_TEMPERATURE if mode == "llm" else 0.0,
        "prompt": prompt_text,
        "raw_response": response.raw if mode == "llm" else EXTRACTIVE_RAW,
        "source_index": response.source_index if mode == "llm" else None,
        "truncated": truncated,
    }

    # --- Stage 11: citations and invariants ------------------------------ #
    trace["stage_11_citation"] = {
        "candidate_url": message.source_url,
        "verified": bool(message.source_url),
        "link_substituted": message.link_substituted,
        "last_updated": message.last_updated,
        "invariants_ok": True,
    }
    citation.assert_invariants(message)

    total_ms = int((time.perf_counter() - started) * 1000)
    trace["timings"] = {
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
    }
    logger.log_query(
        plan,
        result,
        _guardrail_trace(plan, trace),
        message,
        {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "total_ms": total_ms},
    )

    result_out = PipelineResult(message, trace)
    _cache_put(cache_key, result_out, result)
    return result_out


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

#: The in-scope and refutation queries from PRD.md sections 9-10, one per expected
#: outcome. Every one of them must end with a verified link, an in-invariant message
#: and a log line, which is what the smoke test actually asserts. The Parag Parikh
#: queries are *refutations* here because the committed corpus is HDFC-only, and
#: "Can I invest in ..." is refused for advice before the out-of-scope rule is
#: reached - Phase A order is PII, advice, performance, scope.
SMOKE_CASES: tuple[tuple[str, str], ...] = (
    ("What is the exit load on HDFC Large Cap?", "answer"),
    ("What is the expense ratio of HDFC Large Cap?", "answer"),
    ("Minimum SIP for HDFC ELSS Tax Saver Fund", "answer"),
    ("What is the lock-in period on HDFC ELSS Tax Saver Fund?", "answer"),
    ("What is the NAV of HDFC Flexi Cap?", "nav"),
    ("my PAN is AACCK7391J", "pii"),
    ("should I invest in HDFC Flexi Cap?", "advice"),
    ("Can I invest in Parag Parikh Flexi Cap?", "advice"),
    ("How did HDFC Large Cap perform last year?", "performance"),
    ("What is the expense ratio of Mirae Asset Large Cap?", "scope"),
)


def _check_case(result: PipelineResult, expected: str) -> list[str]:
    """Inspect one answered smoke query and return a list of failure descriptions.

    Args:
        result: What :func:`answer` returned.
        expected: The outcome the PRD lists for this query.

    Returns:
        One string per unmet expectation; empty when the case passed.
    """
    failures: list[str] = []
    message = result.message
    try:
        citation.assert_invariants(message)
    except citation.InvariantViolation as exc:  # pragma: no cover - defensive
        failures.append(f"invariant: {exc}")

    # Every refusal renders as kind "refusal"; the expected column says which rule
    # should have produced it, so the message body is checked against that template.
    expected_kind = "answer" if expected == "answer" else "refusal"
    if message.kind != expected_kind:
        failures.append(f"kind: expected {expected_kind}, got {message.kind}")

    if message.source_url and not citation.verify_url(message.source_url):
        failures.append(f"link: {message.source_url!r} is not allow-listed")

    if expected == "answer":
        if not message.source_url:
            failures.append("link: answer has no citation")
        if not message.last_updated:
            failures.append("last_updated: empty")
        if len(config.split_sentences(message.text)) > config.MAX_ANSWER_SENTENCES:
            failures.append("sentences: over the cap")
    else:
        expected_body = config.REFUSAL_TEMPLATES[
            {"pii": "pii", "advice": "advice", "performance": "performance", "scope": "out_of_scope", "nav": "nav"}.get(
                expected, expected
            )
        ]
        if expected_body and expected_body not in message.text:
            failures.append("body: does not match the configured refusal template")
        if re.search(r"\d", message.text):
            failures.append("refusal body contains digits")
    return failures


def run_all(rebuild: bool = False, use_llm: bool | None = None) -> int:
    """Run the stages 5-11 smoke test over the PRD evaluation queries.

    Args:
        rebuild: Re-run ingest first. Off by default so ``run_index.py --stage smoke``
            is offline and quick: the committed snapshot and chunks are what the rest
            of the stages consume, and ``--stage ingest`` exists for re-fetching.
        use_llm: Force generation on or off. ``None`` means "use the LLM if a key is
            configured", which is how the test suite keeps itself offline.

    Returns:
        A process exit code: 0 when every case passed.
    """
    if rebuild:
        from src.ingest import run_all as ingest_all

        ingest_all()

    if use_llm is None:
        use_llm = llm_available()
    elif use_llm:
        config.EXTRACTIVE_ONLY = False

    print(f"[smoke] mode: {'llm' if use_llm else 'extractive (no GROQ_API_KEY)'}")
    clear_cache()
    failures = 0
    for query_text, expected in SMOKE_CASES:
        result = answer(query_text)
        cases = _check_case(result, expected)
        if cases:
            failures += 1
        outcome = f"{result.message.kind:<8} | {result.message.text[:66]}"
        if cases:
            outcome += "  <-- " + "; ".join(cases)
        print(f"[smoke] {expected:<11} {outcome}")

    records = logger.read_log(limit=len(SMOKE_CASES))
    print(f"[smoke] {len(records)} lines in {logger.log_path().name}")
    if failures:
        print(f"[smoke] FAIL: {failures} case(s) did not meet expectations")
    else:
        print("[smoke] PASS: every case returned a verified, in-invariant message")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """``python -m src.pipeline`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Stages 5-11 smoke test")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="use the committed corpus and index instead of rebuilding them",
    )
    parser.add_argument(
        "--extractive",
        action="store_true",
        help="force the extractive path even if GROQ_API_KEY is set",
    )
    args = parser.parse_args(argv)
    return run_all(rebuild=not args.no_rebuild, use_llm=False if args.extractive else None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())