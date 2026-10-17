"""Phase 6 - citation tests (IMPLEMENTATION.md section 6.3).

These tests are the enforcement mechanism for PRD C1 ("every answer cites exactly one
allow-listed source") and C5 ("no hallucinated links"). Nothing here needs a network:
responses are constructed by hand, so the citation layer is provable without an API
key.

The centrepiece is ``test_exactly_one_url_invariant_holds_over_many_samples``, which
counts URLs by regex over 50 rendered messages rather than eyeballing two - a citation
guarantee that is only spot-checked is not a guarantee.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src import config
from src.citation import (
    InvariantViolation,
    URL_PATTERN,
    assert_invariants,
    count_urls,
    finalise,
    finalise_refusal,
    format_last_updated,
    normalize_url,
    render_message,
    resolve_link,
    strip_urls,
    verify_url,
)
from src.types import (
    AssistantMessage,
    LLMResponse,
    RefusalResult,
    RetrievedChunk,
    RetrievalResult,
)

CITATION_SOURCE = Path(__file__).resolve().parent.parent / "src" / "citation.py"

HALLUCINATED_URLS = [
    "https://groww.in.evil.com/mutual-funds/hdfc-large-cap-fund-direct-growth",
    "https://groww.in/mutual-funds/hdfc-parag-parag-fund-direct-growth",
    "https://www.morningstar.in/funds/hdfc-large-cap",
    "https://groww.in/mutual-funds/",
    "https://t.co/abcd1234",
    "http://groww.in.evil.com/mutual-funds/x",
    "https://groww.in.evil.com",
]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_result(count: int = 3, *, scheme: str = "hdfc-large-cap") -> RetrievalResult:
    """A synthetic retrieval result whose chunk URLs are all allow-listed.

    Args:
        count: How many chunks to fabricate.
        scheme: The scheme the chunks belong to.

    Returns:
        A :class:`~src.types.RetrievalResult` with descending scores.
    """
    spec = config.source_spec(scheme) or config.SOURCES[0]
    chunks: list[RetrievedChunk] = []
    for index in range(count):
        chunks.append(
            RetrievedChunk(
                chunk={
                    "chunk_id": f"{scheme}::{index:04d}",
                    "text": f"Exit Load\nExit load of 1% if redeemed within 1 year\nrow {index}",
                    "scheme_key": scheme,
                    "scheme_name": spec.scheme_name,
                    "category": spec.category,
                    "section": "Exit Load",
                    "label": "Exit load",
                    "source_url": spec.url,
                    "source_title": spec.scheme_name,
                    "fetched_at": "2026-09-27",
                },
                score=round(0.7 - index * 0.05, 4),
                rank=index + 1,
            )
        )
    return RetrievalResult(
        chunks=chunks,
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=chunks[0].score,
        used_filter=True,
    )


def make_response(answer: str = "The exit load is 1% if redeemed within 1 year.",
                  source_index: int = 1) -> LLMResponse:
    """A minimal well-formed response for the citation stage."""
    return LLMResponse(
        answer=answer,
        source_index=source_index,
        insufficient_context=False,
        raw='{"answer": "...", "source_index": 1, "insufficient_context": false}',
    )


# --------------------------------------------------------------------------- #
# verify_url - accept
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", sorted(config.ALLOWED_URLS))
def test_verify_url_accepts_every_allow_listed_source(url: str) -> None:
    assert verify_url(url) is True


@pytest.mark.parametrize("url", sorted(config.ALLOWED_URLS))
def test_verify_url_accepts_a_trailing_slash(url: str) -> None:
    assert verify_url(f"{url}/") is True


@pytest.mark.parametrize("url", sorted(config.ALLOWED_URLS))
def test_verify_url_accepts_a_tracking_parameter(url: str) -> None:
    assert verify_url(f"{url}?utm_source=newsletter") is True


@pytest.mark.parametrize("url", sorted(config.ALLOWED_URLS))
def test_verify_url_accepts_http_and_www_variations(url: str) -> None:
    """Normalisation erases the scheme and ``www``, so a plain-http copy of a real
    source is the same page. It is still never what gets rendered: the rendered link
    is always a chunk's own ``https`` URL."""
    with_www = url if "//www." in url else re.sub(r"://", "://www.", url)
    assert verify_url(url.replace("https://", "http://")) is True
    assert verify_url(with_www) is True


def test_verify_url_ignores_case_and_fragment() -> None:
    url = config.SOURCES[0].url
    assert verify_url(url.upper() + "#performance") is True


def test_normalize_url_strips_everything_but_the_path() -> None:
    assert normalize_url("HTTPS://WWW.Groww.in/mutual-funds/x/?utm_source=a#b") == (
        "groww.in/mutual-funds/x"
    )


def test_normalize_url_is_empty_for_junk() -> None:
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""


# --------------------------------------------------------------------------- #
# verify_url - reject
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("url", HALLUCINATED_URLS)
def test_verify_url_rejects_look_alikes(url: str) -> None:
    """groww.in.evil.com, unknown slugs, non-Groww AMC URLs, shorteners."""
    assert verify_url(url) is False


def test_verify_url_rejects_a_different_amc_page() -> None:
    assert verify_url("https://www.hdfcmf.com/hdfc-large-cap") is False


def test_verify_url_rejects_the_bare_allow_list_base() -> None:
    """The base is a prefix, not a source."""
    assert verify_url(config.ALLOWED_URL_BASE) is False


@pytest.mark.parametrize("url", ["", "   ", "not a url", "https://"])
def test_verify_url_rejects_empty_and_malformed(url: str) -> None:
    assert verify_url(url) is False


def test_a_look_alike_host_is_not_matched_by_a_substring_test() -> None:
    """The reason verification normalises before comparing rather than using
    ``ALLOWED_URL_BASE in url``."""
    assert config.ALLOWED_URL_BASE not in "https://groww.in.evil.com/mutual-funds/x"


# --------------------------------------------------------------------------- #
# resolve_link
# --------------------------------------------------------------------------- #


def test_resolve_link_uses_the_cited_block() -> None:
    result = make_result()
    url, substituted = resolve_link(make_response(source_index=2), result)
    assert url == result.chunks[1].source_url
    assert substituted is False


def test_resolve_link_falls_back_to_the_top_chunk_for_a_bad_index() -> None:
    result = make_result()
    for bad_index in (0, -1, 99, None):
        url, substituted = resolve_link(make_response(source_index=bad_index), result)
        assert url == result.chunks[0].source_url
        assert substituted is False


def test_resolve_link_discards_an_unverified_block_url() -> None:
    """A hallucinated source URL is dropped, never repaired."""
    result = make_result()
    tampered = RetrievedChunk(
        chunk={**result.chunks[1].chunk, "source_url": "https://groww.in.evil.com/funds/x"},
        score=result.chunks[1].score,
        rank=2,
    )
    tampered_result = RetrievalResult(
        chunks=[result.chunks[0], tampered],
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=result.chunks[0].score,
        used_filter=True,
    )
    url, substituted = resolve_link(make_response(source_index=2), tampered_result)
    assert url == result.chunks[0].source_url
    assert substituted is True
    assert verify_url(url)


def test_resolve_link_returns_empty_when_nothing_was_retrieved() -> None:
    empty = RetrievalResult([], 0, 0, 0.0, False)
    assert resolve_link(make_response(), empty) == ("", False)


# --------------------------------------------------------------------------- #
# finalise
# --------------------------------------------------------------------------- #


def test_finalise_produces_a_complete_answer() -> None:
    result = make_result()
    message = finalise(make_response(), result)
    assert message.kind == "answer"
    assert message.source_url == result.chunks[0].source_url
    assert message.last_updated == "27 Sep 2026"
    assert message.disclaimer == config.DISCLAIMER
    assert message.truncated is False
    assert message.link_substituted is False


def test_finalise_takes_the_freshness_date_from_chunk_metadata() -> None:
    """An LLM that invents a date must not be able to set the stamp."""
    result = make_result()
    answer = "The lock-in period is 3 years, as updated on 01 Jan 2020."
    message = finalise(make_response(answer=answer), result)
    assert message.last_updated == "27 Sep 2026"
    assert "Last updated from sources: 27 Sep 2026" in render_message(message)


def test_hallucinated_url_in_the_prose_is_stripped_and_flagged() -> None:
    result = make_result()
    answer = "The exit load is 1%. Source: https://groww.in.evil.com/mutual-funds/x"
    message = finalise(make_response(answer=answer), result)
    assert message.link_substituted is True
    assert count_urls(message.text) == 0
    rendered = render_message(message)
    assert "groww.in.evil.com" not in rendered
    assert count_urls(rendered) == 1
    assert verify_url(result.chunks[0].source_url)


def test_sentence_cap_truncates_and_flags() -> None:
    result = make_result()
    six = (
        "One. Two. Three. Four. Five. Six."
    )
    message = finalise(make_response(answer=six), result)
    assert config.sentence_count(message.text) == config.MAX_ANSWER_SENTENCES
    assert message.text.startswith("One. Two. Three.")
    assert message.truncated is True


def test_a_short_answer_is_not_marked_truncated() -> None:
    message = finalise(make_response(answer="Only one sentence."), make_result())
    assert message.truncated is False


@pytest.mark.parametrize("index", [1, 2, 3])
def test_finalise_survives_every_valid_source_index(index: int) -> None:
    message = finalise(make_response(source_index=index), make_result())
    assert verify_url(message.source_url)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", sorted(config.REFUSAL_TEMPLATES))
def test_finalise_refusal_renders_a_refusal_block(kind: str) -> None:
    refusal = RefusalResult(
        kind=kind,
        message=config.REFUSAL_TEMPLATES[kind],
        link=config.default_link(),
        rule=f"{kind}.*",
    )
    message = finalise_refusal(refusal)
    assert message.kind == "refusal"
    assert message.source_url == config.default_link()
    assert message.last_updated == ""
    rendered = render_message(message)
    assert rendered.startswith(config.REFUSAL_TEMPLATES[kind])
    assert config.DISCLAIMER in rendered
    assert "Source: " not in rendered, "a refusal shows no citation badge"


# --------------------------------------------------------------------------- #
# assert_invariants
# --------------------------------------------------------------------------- #


def _message(**overrides: object) -> AssistantMessage:
    """A valid answer, with fields overridable to break one invariant at a time."""
    base = {
        "kind": "answer",
        "text": "The exit load is 1% if redeemed within 1 year.",
        "source_url": config.SOURCES[0].url,
        "last_updated": "27 Sep 2026",
        "disclaimer": config.DISCLAIMER,
    }
    base.update(overrides)
    return AssistantMessage(**base)  # type: ignore[arg-type]


def test_assert_invariants_accepts_a_valid_message() -> None:
    assert_invariants(_message())


def test_two_urls_are_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(
            _message(text="Exit load is 1% at https://groww.in/mutual-funds/x.")
        )
    assert excinfo.value.invariant == "a.exactly_one_url"


def test_an_unverified_url_is_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(_message(source_url="https://groww.in.evil.com/x"))
    assert excinfo.value.invariant == "b.url_allow_listed"


def test_a_missing_disclaimer_is_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(_message(disclaimer=""))
    assert excinfo.value.invariant == "d.disclaimer_present"


def test_a_missing_last_updated_line_is_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(_message(last_updated=""))
    assert excinfo.value.invariant == "c.last_updated_present"


def test_a_refusal_containing_digits_is_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(
            AssistantMessage(
                kind="refusal",
                text="I can't share returns for 3 years.",
                source_url=config.SOURCES[0].url,
                last_updated="",
                disclaimer=config.DISCLAIMER,
            )
        )
    assert excinfo.value.invariant == "f.refusal_has_no_digits"


def test_a_body_over_the_sentence_cap_is_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(_message(text="One. Two. Three. Four."))
    assert excinfo.value.invariant == "e.sentence_cap"


def test_a_url_in_the_body_is_rejected_even_with_a_valid_source_url() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(
            _message(text="See https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth")
        )
    assert excinfo.value.invariant in {"a.exactly_one_url", "g.body_has_no_url"}


def test_a_message_with_no_link_at_all_is_rejected() -> None:
    with pytest.raises(InvariantViolation) as excinfo:
        assert_invariants(_message(source_url=""))
    assert excinfo.value.invariant == "a.exactly_one_url"


# --------------------------------------------------------------------------- #
# The one-link guarantee, counted rather than eyeballed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index", range(50))
def test_exactly_one_url_invariant_holds_over_many_samples(index: int, chunks) -> None:
    """Build 50 rendered messages from real corpus chunks and count URLs in each."""
    if not chunks:
        pytest.skip("chunks.jsonl not built")
    corpus_chunk = chunks[index % len(chunks)]
    retrieved = RetrievedChunk(chunk=corpus_chunk, score=0.7, rank=1)
    result = RetrievalResult([retrieved], 0, 0, 0.7, True)
    answers = [
        f"The value for {corpus_chunk['label']} is {corpus_chunk['chunk_id']}.",
        "Exit load of 1% if redeemed within 1 year.",
        "Source: https://example.com/not-allowed",
        "One. Two. Three. Four. Five.",
        "Expense ratio 1.04%. NAV: \u20b91,151.92. Min. for SIP \u20b9100.",
    ]
    answer = answers[index % len(answers)]
    response = LLMResponse(
        answer=answer,
        source_index=1 + (index % 3),
        insufficient_context=False,
        raw="{}",
    )
    message = finalise(response, result)
    rendered = render_message(message)
    assert count_urls(rendered) == 1, rendered
    assert verify_url(message.source_url)
    assert config.DISCLAIMER in rendered


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_strip_urls_removes_a_trailing_period_correctly() -> None:
    cleaned, had_url = strip_urls("Exit load is 1%. Source: https://x.example/y.")
    assert had_url is True
    assert "x.example" not in cleaned
    assert cleaned.endswith(".")


def test_strip_urls_is_a_no_op_on_link_free_text() -> None:
    cleaned, had_url = strip_urls("Expense ratio is 1.04%.")
    assert had_url is False
    assert cleaned == "Expense ratio is 1.04%."


def test_strip_urls_handles_empty_text() -> None:
    assert strip_urls("") == ("", False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2026-09-27", "27 Sep 2026"), ("2026-01-05", "5 Jan 2026"), ("", ""), ("junk", "")],
)
def test_format_last_updated(raw: str, expected: str) -> None:
    assert format_last_updated(raw) == expected


def test_render_message_puts_the_disclaimer_last() -> None:
    lines = render_message(_message()).splitlines()
    assert lines[-1] == config.DISCLAIMER
    assert lines[-2].startswith(config.LAST_UPDATED_PREFIX)
    assert lines[-3].startswith("Source: ")


def test_count_urls_matches_the_regex() -> None:
    text = "a https://x.example b http://y.example"
    assert count_urls(text) == len(URL_PATTERN.findall(text)) == 2


def test_citation_module_imports_no_llm_and_no_retriever() -> None:
    """A citation decision must be reachable with no model and no index."""
    tree = ast.parse(CITATION_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = sorted(
        name
        for name in imported
        if any(word in name for word in ("generator", "groq", "retriever", "langchain"))
    )
    assert offenders == [], f"citation.py must not import {offenders}"
    assert re.match(r"^\s*\"\"\"", CITATION_SOURCE.read_text(encoding="utf-8"))