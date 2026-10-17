"""Phase 7 - logging and pipeline tests (IMPLEMENTATION.md section 7.1).

The pipeline is tested with the *retriever faked out*: a stub result built from real
chunk metadata is enough to drive every branch, and it keeps the suite offline and
fast. The one test that runs the true retriever is marked as the integration check -
that is where "the stages are wired to each other, not just to their own doubles"
gets proven.

No test here needs ``GROQ_API_KEY``. Generation is stubbed for the llm-path tests and
switched off for the rest, which is the same degradation the demo relies on.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from src import citation, config, guardrails, logger, pipeline
from src.generator import LLMUnavailable, ParseFailure
from src.types import AssistantMessage, LLMResponse, RetrievalResult, RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
PAN = "AACCK7391J"

TRACE_KEYS = {
    "stage_5_rewrite",
    "guardrails_a",
    "stage_6_8_retrieval",
    "guardrails_b",
    "stage_10_generation",
    "stage_11_citation",
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _chunk(
    text: str,
    label: str,
    score: float,
    rank: int,
    *,
    section: str = "HDFC Large Cap Fund — Direct Growth",
    scheme_key: str = "hdfc-large-cap",
    url: str | None = None,
    fetched_at: str = "2026-09-27",
) -> RetrievedChunk:
    """Build a chunk the citation layer will accept.

    The URL defaults to the scheme's configured one: the allow-list is exact, so a
    made-up path would make every assertion below fail for the wrong reason.
    """
    spec = next(s for s in config.SOURCES if s.scheme_key == scheme_key)
    return RetrievedChunk(
        chunk={
            "text": text,
            "label": label,
            "section": section,
            "scheme_key": scheme_key,
            "scheme_name": spec.scheme_name,
            "source_url": url or spec.url,
            "fetched_at": fetched_at,
        },
        score=score,
        rank=rank,
    )


EXIT_LOAD_CHUNK = _chunk(
    "HDFC Large Cap Fund Direct Growth is a Equity Mutual Fund Scheme launched by HDFC "
    "Mutual Fund. This scheme was made available to investors on 10 Dec 1999. Rahul Baijal "
    "is the Current Fund Manager of HDFC Large Cap Fund Direct Growth fund. The HDFC Large "
    "Cap Fund Direct Growth is rated Very High risk. Minimum SIP Investment is set to ₹100. "
    "Minimum Lumpsum Investment is ₹100. Exit load of 1% if redeemed within 1 year",
    "Exit load",
    0.62,
    1,
)

EXPENSE_CHUNK = _chunk(
    "# HDFC Large Cap Fund Direct Growth\n\nEquity\nLarge Cap\nVery High Risk\n"
    "NAV: 01 Oct '26\n₹1,151.92\nMin. for SIP\n₹100\nFund size (AUM)\n₹39,933.36 Cr\n"
    "Expense ratio\n1.04%\nRating\n4",
    "Expense ratio",
    0.55,
    2,
)

UNRELATED_CHUNK = _chunk(
    "HDFC Large Cap Fund Direct Growth is rated Very High risk.",
    "HDFC Large Cap Fund — Direct Growth",
    0.44,
    3,
)

MANAGER_CHUNK = _chunk(
    "### DM Dhruv Muchhal Jun 2023 - Present View details\n\nEducation\nMr. Dhruv has "
    "done B.Com, CA and CFA\nExperience\nPrior to joining HDFC MF,he has worked with "
    "Motilal Oswal Financial Services Ltd., Goldman Sachs, CRISIL Global Research & "
    "Analytics and Deloitte Haskins & Sells",
    "Fund management",
    0.92,
    1,
)

SECOND_MANAGER_CHUNK = _chunk(
    "### RB Rahul Baijal Jul 2022 - Present View details\n\nEducation\nMr. Baijal has "
    "done PGDM(MBA) from IIM Calcutta and is an engineering graduate of Delhi college "
    "of Engineering\nExperience\nPrior to joining HDFC Mutual Fund, he has worked with "
    "Sundaram Mutual Fund, Bharti AXA Life Insurance, TVF Capital, HSBC Securities, "
    "Credit Suisse Securities & Standard Chartered Bank.",
    "Fund management",
    0.91,
    2,
)


@pytest.fixture
def fake_retrieval(monkeypatch: pytest.MonkeyPatch) -> RetrievalResult:
    """Answer every query with three deterministic chunks, no model involved."""
    result = RetrievalResult(
        chunks=[EXIT_LOAD_CHUNK, EXPENSE_CHUNK, UNRELATED_CHUNK],
        dropped_low_score=2,
        dropped_scheme_mismatch=0,
        confidence=0.62,
        used_filter=True,
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda plan: result)
    return result


@pytest.fixture(autouse=True)
def offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No API key, no shared cache, and a log file inside the test's tmp dir."""
    monkeypatch.setitem(config.PATHS, "LOG_FILE", tmp_path / "query_log.jsonl")
    monkeypatch.setattr(pipeline, "llm_available", lambda: False)
    pipeline.clear_cache()


# --------------------------------------------------------------------------- #
# Extractive path
# --------------------------------------------------------------------------- #


def test_extractive_answer_is_verified_and_dated(fake_retrieval: RetrievalResult) -> None:
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    message = result.message
    assert message.kind == "answer"
    assert citation.verify_url(message.source_url)
    assert message.last_updated == "27 Sep 2026"
    assert len(config.split_sentences(message.text)) <= config.MAX_ANSWER_SENTENCES


def test_extractive_body_quotes_the_chunk_it_came_from(
    fake_retrieval: RetrievalResult,
) -> None:
    """Every number in an extractive answer must already be in the retrieved text."""
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    corpus = " ".join(retrieved.chunk["text"] for retrieved in fake_retrieval.chunks)
    for number in re.findall(r"[\d.]+", result.message.text):
        assert number in corpus


def test_extractive_picks_the_chunk_matching_the_intent(
    fake_retrieval: RetrievalResult,
) -> None:
    """An expense-ratio question must not be answered from the exit-load chunk."""
    result = pipeline.answer("What is the expense ratio of HDFC Large Cap?")
    assert "1.04%" in result.message.text


def test_extractive_fund_manager_answer_names_the_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manager's name lives in the heading, which _clean_text drops.

    The extractive body must still surface it, or the degraded path answers a
    fund-manager question with the bio's experience sentence and no name.
    """
    result = RetrievalResult(
        chunks=[MANAGER_CHUNK],
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=0.92,
        used_filter=True,
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda plan: result)
    message = pipeline.answer("who is the fund manager of HDFC Large Cap").message
    assert message.kind == "answer"
    assert "Dhruv Muchhal" in message.text
    assert "Jun 2023" in message.text


def test_extractive_fund_manager_answer_names_every_retrieved_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corpus gives each manager their own chunk.

    A body built from the single picked chunk would name only the first manager
    and silently drop the second one a scheme legitimately has.
    """
    result = RetrievalResult(
        chunks=[MANAGER_CHUNK, SECOND_MANAGER_CHUNK],
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=0.92,
        used_filter=True,
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda plan: result)
    message = pipeline.answer("who is the fund manager of HDFC Large Cap").message
    assert "Dhruv Muchhal" in message.text
    assert "Rahul Baijal" in message.text
    # One field, both managers, tenure bracketed rather than run on after the name.
    assert message.text.count("Fund management:") == 1
    assert "(Jun 2023 - Present)" in message.text
    assert "(Jul 2022 - Present)" in message.text
    assert len(config.split_sentences(message.text)) <= config.MAX_ANSWER_SENTENCES


def test_extractive_min_sip_answer_quotes_only_the_asked_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheme card stacks the SIP and lump-sum rows under one heading.

    A min-SIP question must quote the SIP row and say which fund it is for,
    not paste every row of the card.
    """
    card = _chunk(
        "### Minimum investments\n\nMin. for 1st investment\n₹500\nMin. for 2nd investment\n"
        "₹500\nMin. for SIP\n₹500",
        "Minimum SIP",
        0.93,
        1,
        scheme_key="hdfc-elss-tax-saver",
    )
    result = RetrievalResult(
        chunks=[card],
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=0.93,
        used_filter=True,
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda plan: result)
    message = pipeline.answer("min sip for elss?").message
    spec = next(s for s in config.SOURCES if s.scheme_key == "hdfc-elss-tax-saver")
    assert message.text.startswith(spec.scheme_name)
    assert "Min. for SIP" in message.text
    assert "₹500" in message.text
    assert "1st investment" not in message.text


def test_extractive_never_exceeds_the_sentence_cap(fake_retrieval: RetrievalResult) -> None:
    for query_text in (
        "What is the exit load on HDFC Large Cap?",
        "Minimum SIP for HDFC Large Cap",
        "What is the risk rating of HDFC Large Cap?",
    ):
        result = pipeline.answer(query_text)
        assert len(config.split_sentences(result.message.text)) <= config.MAX_ANSWER_SENTENCES


def test_answer_without_a_key_uses_the_extractive_path(
    fake_retrieval: RetrievalResult,
) -> None:
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    generation = result.trace["stage_10_generation"]
    assert generation["mode"] == pipeline.EXTRACTIVE_MODE
    assert "GROQ_API_KEY" in generation["reason"]
    assert result.message.cached is False


def test_extractive_only_mode_never_calls_the_model(
    fake_retrieval: RetrievalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(plan, result):  # noqa: ANN001, ANN202 - must not be called
        raise AssertionError("generate() called while EXTRACTIVE_ONLY is set")

    # A key *is* configured, so the only thing that can keep the model away is the
    # flag: that is the behaviour worth pinning down.
    monkeypatch.setattr(config, "EXTRACTIVE_ONLY", True)
    monkeypatch.setattr(config, "load_api_key", lambda required=False: "test-key")
    monkeypatch.setattr(pipeline, "generate", explode)
    assert pipeline.llm_available() is False
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert result.trace["stage_10_generation"]["mode"] == pipeline.EXTRACTIVE_MODE
    assert "EXTRACTIVE_ONLY" in result.trace["stage_10_generation"]["reason"]


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "error",
    [
        LLMUnavailable("429 rate limited"),
        ParseFailure("garbage"),
        RuntimeError("network down"),
    ],
    ids=["unavailable", "parse-failure", "unexpected"],
)
def test_a_failing_model_degrades_to_the_extractive_path(
    fake_retrieval: RetrievalResult,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """A 429 from a free-tier key must not turn the demo into a stack trace."""

    def boom(plan, result):  # noqa: ANN001, ANN202 - stub
        raise error

    monkeypatch.setattr(pipeline, "llm_available", lambda: True)
    monkeypatch.setattr(pipeline, "generate", boom)
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert result.message.kind == "answer"
    assert result.trace["stage_10_generation"]["mode"] == pipeline.EXTRACTIVE_MODE
    citation.assert_invariants(result.message)


def test_a_model_answer_is_used_when_it_parses(
    fake_retrieval: RetrievalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "llm_available", lambda: True)
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda plan, result: LLMResponse(
            answer="The exit load is 1% in the first year.",
            source_index=1,
            insufficient_context=False,
            raw='{"answer": "The exit load is 1% in the first year.", "source_index": 1}',
        ),
    )
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert result.message.text == "The exit load is 1% in the first year."
    assert result.trace["stage_10_generation"]["mode"] == "llm"
    assert result.trace["stage_10_generation"]["raw_response"].startswith('{"answer"')


def test_a_hallucinated_url_is_replaced_by_the_chunks(
    fake_retrieval: RetrievalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "llm_available", lambda: True)
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda plan, result: LLMResponse(
            answer="See https://groww.in/elss-tax-saver-fund for the scheme.",
            source_index=1,
            insufficient_context=False,
            raw="see https://groww.in/elss-tax-saver-fund",
        ),
    )
    message = pipeline.answer("What is the exit load on HDFC Large Cap?").message
    assert citation.verify_url(message.source_url)
    assert message.link_substituted is True
    assert "groww.in/elss-tax-saver-fund" not in message.text
    citation.assert_invariants(message)


def test_insufficient_context_becomes_a_low_confidence_refusal(
    fake_retrieval: RetrievalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "llm_available", lambda: True)
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda plan, result: LLMResponse(
            answer="",
            source_index=1,
            insufficient_context=True,
            raw='{"answer": "", "insufficient_context": true}',
        ),
    )
    result = pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert result.message.kind == "refusal"
    assert result.message.last_updated == ""
    assert result.trace["stage_10_generation"]["insufficient_context"] is True


# --------------------------------------------------------------------------- #
# Guardrails inside the pipeline
# --------------------------------------------------------------------------- #


def test_a_pii_query_never_reaches_the_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(plan):  # noqa: ANN001 - must not be called
        raise AssertionError("retrieval ran on a PII query")

    monkeypatch.setattr(pipeline, "retrieve", explode)
    result = pipeline.answer(f"my PAN is {PAN}, what are my returns?")
    assert result.message.kind == "refusal"
    assert "stage_6_8_retrieval" not in result.trace


def test_the_trace_never_leaks_the_pan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "retrieve", lambda plan: RetrievalResult([], 0, 0, 0.0, False))
    result = pipeline.answer(f"my PAN is {PAN}, what are my returns?")
    assert PAN not in json.dumps(result.trace)
    assert "<pii:pan>" in result.trace["stage_5_rewrite"]["raw"]


@pytest.mark.parametrize(
    "query_text,expected_template",
    [
        (f"my PAN is {PAN}", "pii"),
        ("should I invest in HDFC Flexi Cap?", "advice"),
        ("How did HDFC Large Cap perform last year?", "performance"),
        ("What is the expense ratio of Mirae Asset Large Cap?", "out_of_scope"),
        ("What is the NAV of HDFC Flexi Cap?", "nav"),
    ],
)
def test_every_refusal_path_is_verified_and_digitless(
    fake_retrieval: RetrievalResult, query_text: str, expected_template: str
) -> None:
    message = pipeline.answer(query_text).message
    assert message.kind == "refusal"
    assert message.last_updated == ""
    citation.assert_invariants(message)
    assert citation.verify_url(message.source_url)
    assert config.REFUSAL_TEMPLATES[expected_template] in message.text
    assert not re.search(r"\d", message.text)


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #


def test_the_trace_always_carries_every_stage_key(
    fake_retrieval: RetrievalResult,
) -> None:
    trace = pipeline.answer("What is the exit load on HDFC Large Cap?").trace
    assert TRACE_KEYS <= set(trace)


def test_a_phase_a_refusal_reports_only_the_stages_it_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrieval never ran, so it is absent - not faked as an empty result."""
    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda plan: pytest.fail("retrieval ran on a PII query"),
    )
    trace = pipeline.answer(f"my PAN is {PAN}").trace
    assert set(trace) == {"stage_5_rewrite", "guardrails_a", "guardrails_b", "stage_11_citation"}
    assert trace["guardrails_b"] == []


def test_the_rewrite_stage_shows_raw_normalized_and_expanded(
    fake_retrieval: RetrievalResult,
) -> None:
    trace = pipeline.answer("What is the exit load on HDFC Large Cap?").trace
    rewrite = trace["stage_5_rewrite"]
    assert rewrite["raw"].startswith("What is the exit load")
    assert rewrite["normalized"]
    assert rewrite["expanded"]
    assert rewrite["intent"]


def test_the_retrieval_stage_lists_what_survived(fake_retrieval: RetrievalResult) -> None:
    trace = pipeline.answer("What is the exit load on HDFC Large Cap?").trace
    retrieval = trace["stage_6_8_retrieval"]
    assert retrieval["top_k"] == config.TOP_K
    assert retrieval["floor"] > 0
    assert len(retrieval["chunks"]) == len(fake_retrieval.chunks)
    assert retrieval["chunks"][0]["rank"] == 1
    assert all(chunk["preview"] for chunk in retrieval["chunks"])


def test_the_citation_stage_records_the_link_decision(
    fake_retrieval: RetrievalResult,
) -> None:
    trace = pipeline.answer("What is the exit load on HDFC Large Cap?").trace
    citation_trace = trace["stage_11_citation"]
    assert citation_trace["verified"] is True
    assert citation_trace["last_updated"] == "27 Sep 2026"
    assert citation_trace["invariants_ok"] is True


def test_both_guardrail_stages_appear_in_the_trace(
    fake_retrieval: RetrievalResult,
) -> None:
    trace = pipeline.answer("What is the exit load on HDFC Large Cap?").trace
    assert [entry["rule"] for entry in trace["guardrails_a"]]
    assert [entry["rule"] for entry in trace["guardrails_b"]]
    assert all("hit" in entry for entry in trace["guardrails_b"])


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_the_second_identical_query_comes_from_the_cache_in_under_50ms(
    fake_retrieval: RetrievalResult,
) -> None:
    import time

    first = pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert first.message.cached is False

    started = time.perf_counter()
    second = pipeline.answer("What is the exit load on HDFC Large Cap?")
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert second.message.cached is True
    assert second.message.text == first.message.text
    assert second.message.source_url == first.message.source_url
    assert elapsed_ms < 50
    assert second.trace["cached"] is True


def test_the_cache_key_is_the_expanded_query_not_the_raw_one(
    fake_retrieval: RetrievalResult,
) -> None:
    pipeline.answer("exit load on HDFC Large Cap")
    again = pipeline.answer("  EXIT   LOAD on HDFC Large Cap!  ")
    assert again.message.cached is True


def test_a_refused_query_is_never_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def counting(plan):  # noqa: ANN001 - stub
        calls["n"] += 1
        return RetrievalResult([], 0, 0, 0.0, False)

    monkeypatch.setattr(pipeline, "retrieve", counting)
    for _ in range(2):
        assert pipeline.answer(f"my PAN is {PAN}").message.kind == "refusal"
    assert calls["n"] == 0


def test_the_cache_expires_after_the_ttl(
    fake_retrieval: RetrievalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.pipeline as module

    clock = {"now": 1_000.0}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])
    pipeline.answer("What is the exit load on HDFC Large Cap?")

    clock["now"] += config.CACHE_TTL_SECONDS + 1
    assert pipeline.answer("What is the exit load on HDFC Large Cap?").message.cached is False


def test_clear_cache_empties_it(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert pipeline.clear_cache() == 1
    assert pipeline.answer("What is the exit load on HDFC Large Cap?").message.cached is False


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_every_query_writes_exactly_one_line(fake_retrieval: RetrievalResult) -> None:
    before = len(logger.read_log())
    for query_text in (
        "What is the exit load on HDFC Large Cap?",
        "Minimum SIP for HDFC Large Cap",
        f"my PAN is {PAN}",
    ):
        pipeline.answer(query_text)
    assert len(logger.read_log()) == before + 3


def test_a_pii_query_is_logged_as_a_placeholder(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer(f"my PAN is {PAN}, what are my returns?")
    line = logger.read_log()[-1]
    assert line["query_redacted"] == config.PII_LOG_PLACEHOLDER
    assert line["pii_detected"] is True
    assert PAN not in json.dumps(line)


def test_no_log_file_ever_contains_a_pan(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer(f"share my aadhaar {PAN} number")
    pipeline.answer(f"my PAN is {PAN}")
    assert PAN not in logger.log_path().read_text(encoding="utf-8")


def test_a_log_line_has_the_documented_fields(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer("What is the exit load on HDFC Large Cap?")
    line = logger.read_log()[-1]
    for field in (
        "ts",
        "session",
        "query_redacted",
        "pii_detected",
        "intent",
        "refusal",
        "top_score",
        "sources",
        "chunks_used",
        "cached",
        "llm_ms",
        "total_ms",
    ):
        assert field in line
    assert line["session"] == "anon"
    assert line["intent"] == "exit_load"
    assert line["refusal"] == []
    assert line["chunks_used"] == len(line["sources"]) == 3
    assert citation.verify_url(line["sources"][0])
    assert line["cached"] is False


def test_the_refusal_rule_is_logged(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer("should I invest in HDFC Flexi Cap?")
    assert "advice.recommend" in logger.read_log()[-1]["refusal"]


def test_a_cache_hit_is_logged_as_one(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer("What is the exit load on HDFC Large Cap?")
    pipeline.answer("What is the exit load on HDFC Large Cap?")
    assert [line["cached"] for line in logger.read_log()[-2:]] == [False, True]


def test_read_log_respects_the_limit_and_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert logger.read_log() == []
    monkeypatch.setitem(config.PATHS, "LOG_FILE", tmp_path / "nested" / "absent.jsonl")
    assert logger.read_log() == []
    logger.log_query(
        _plan("What is the exit load on HDFC Large Cap?"),
        None,
        [],
        AssistantMessage(
            kind="refusal", text="No.", source_url="", last_updated="", disclaimer=config.DISCLAIMER
        ),
    )
    assert logger.log_path().exists()
    assert len(logger.read_log(limit=1)) == 1


def test_a_corrupt_line_is_skipped(fake_retrieval: RetrievalResult) -> None:
    pipeline.answer("What is the exit load on HDFC Large Cap?")
    with logger.log_path().open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    assert len(logger.read_log()) == 1


@pytest.mark.parametrize(
    "text,marker",
    [
        (f"my PAN is {PAN}", "<pii:pan>"),
        ("email me at ravi.sharma@example.com", "<pii:email>"),
        ("call me on +91 98765 43210", "<pii:phone>"),
        ("otp is 123456", "<pii:otp>"),
    ],
)
def test_redact_replaces_every_pii_kind(text: str, marker: str) -> None:
    redacted = logger.redact(text)
    assert marker in redacted
    assert "ravi.sharma@example.com" not in redacted


def test_redact_leaves_a_clean_query_alone() -> None:
    assert logger.redact("What is the exit load on HDFC Large Cap?") == (
        "What is the exit load on HDFC Large Cap?"
    )
    assert logger.redact("") == ""


def test_redact_is_a_second_layer_even_when_the_guardrail_did_not_fire() -> None:
    """Even a query the guardrails let through cannot reach the log intact."""
    assert PAN not in logger.redact(f"scheme detail for {PAN}")


# --------------------------------------------------------------------------- #
# Wiring and smoke
# --------------------------------------------------------------------------- #


def _plan(query_text: str):
    from src import query

    return query.build_plan(query_text)


def test_nothing_in_the_package_imports_the_ui() -> None:
    """Phase 8's rule, asserted now so it cannot be broken silently later."""
    offenders = []
    for path in sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "app" for name in names):
                offenders.append(path.name)
    assert offenders == []


def test_the_smoke_run_passes_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The committed corpus must satisfy every PRD evaluation query with no key."""
    assert pipeline.run_all(rebuild=False, use_llm=False) == 0
    out = capsys.readouterr().out
    assert "[smoke] PASS" in out
    for _query_text, expected in pipeline.SMOKE_CASES:
        assert expected in out


@pytest.mark.slow
def test_the_real_retriever_is_wired_up() -> None:
    """Integration: the real index, the real embedding model, no key required."""
    pipeline.clear_cache()
    result = pipeline.answer("What is the expense ratio of HDFC Large Cap?")
    citation.assert_invariants(result.message)
    assert result.message.kind == "answer"
    assert citation.verify_url(result.message.source_url)
    assert result.message.last_updated
    assert result.trace["stage_6_8_retrieval"]["chunks"]
    assert logger.read_log()[-1]["intent"] == "expense_ratio"