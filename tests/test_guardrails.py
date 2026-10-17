"""Phase 5 - guardrail tests (IMPLEMENTATION.md section 5.3).

The largest test file in the project, and deliberately so: the refusal policy is the
part of this assistant a user can be harmed by if it is wrong, so it is the part
that is pinned hardest.

Four properties are enforced here:

* **Correctness** - parametrised positive/negative samples per rule, with the
  negatives drawn from the PRD section 12 factual queries. A false positive on
  "what is the expense ratio" would refuse every normal question, so those must stay
  silent.
* **Priority** - ``pii`` beats ``advice`` on a message that contains both, which is
  what keeps personal data out of the retrieval step and out of the log.
* **Shape** - every refusal is <= 3 sentences, carries no digits, carries exactly one
  allow-listed link, and contains no URL of its own.
* **Purity** - this module must never import the generator or the retriever, so a
  refusal can be produced with no model, no index and no network.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src import config, query
from src.guardrails import (
    SCOPE_RULE,
    check_phase_a,
    check_phase_b,
    describe_phase_a,
    describe_phase_b,
    detect_advice,
    detect_performance,
    detect_pii,
    detect_scope,
    educational_link,
    expected_labels,
    grounded,
    render_refusal,
)
from src.types import RetrievedChunk, RetrievalResult

#: Rule ids that must each have positive and negative samples.
EXPECTED_RULES = [
    "pii.pan",
    "pii.aadhaar",
    "pii.account",
    "pii.email",
    "pii.phone",
    "pii.otp",
    "advice.recommend",
    "advice.trade",
    "advice.advice_seek",
    "performance.compare",
]

GUARDRAILS_SOURCE = Path(__file__).resolve().parent.parent / "src" / "guardrails.py"


# --------------------------------------------------------------------------- #
# PII - the six shapes from PRD C2
# --------------------------------------------------------------------------- #

PII_POSITIVES: list[tuple[str, str]] = [
    ("pii.pan", "My PAN is ABCDE1234F, can you check my SIP?"),
    ("pii.pan", "pan: aacck7391j"),
    ("pii.pan", "PAN number - ZZZZZ1111A - what is the exit load?"),
    ("pii.aadhaar", "My Aadhaar is 1234 5678 9012"),
    ("pii.aadhaar", "aadhaar 123456789012"),
    ("pii.aadhaar", "Aadhaar number 1234-5678-9012"),
    ("pii.account", "Account number 1234567890"),
    ("pii.account", "my folio is 00123456789"),
    ("pii.email", "email me at manliv@example.com"),
    ("pii.email", "first.last+tag@sub.domain.co.in"),
    ("pii.phone", "call me on +91 98765 43210"),
    ("pii.phone", "reach me on 98765 43210"),
    ("pii.otp", "my otp is 123456"),
    ("pii.otp", "send the one time password"),
    ("pii.otp", "the verification code is 4412"),
]

PII_NEGATIVES: list[str] = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "expense ratio is 0.52% for the direct plan",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "What is the minimum SIP for the direct plan?",
    "What is the exit load after 12 months?",
    "hdfc-elss",
    "the scheme was launched in 1999",
    "Is there an exit load of 1 percent?",
    "What is the minimum lump sum of HDFC Small Cap?",
    "What benchmark does HDFC Balanced Advantage track?",
    "What is the riskometer level of HDFC Large Cap?",
]


@pytest.mark.parametrize(("rule", "text"), PII_POSITIVES)
def test_pii_detector_fires(rule: str, text: str) -> None:
    assert rule in detect_pii(text), f"{rule} missed: {text}"


@pytest.mark.parametrize("text", PII_NEGATIVES)
def test_pii_detector_stays_silent_on_factual_queries(text: str) -> None:
    assert detect_pii(text) == [], text


# --------------------------------------------------------------------------- #
# Advice
# --------------------------------------------------------------------------- #

ADVICE_POSITIVES: list[tuple[str, str]] = [
    ("advice.recommend", "Should I buy HDFC Small Cap?"),
    ("advice.recommend", "shall i redeem after 3 years?"),
    ("advice.trade", "How should I allocate 60% to small cap?"),
    ("advice.trade", "should I sell after 3 years"),
    ("advice.advice_seek", "Which fund is best for me?"),
    ("advice.advice_seek", "Is now a good time to invest?"),
    ("advice.advice_seek", "Would you recommend the flexi cap fund?"),
    ("advice.advice_seek", "Is HDFC Small Cap good for me?"),
    ("advice.advice_seek", "Can you suggest a portfolio?"),
]

ADVICE_NEGATIVES: list[str] = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "What is the minimum lump sum for HDFC Small Cap?",
    "What benchmark does HDFC Balanced Advantage track?",
    "What is the riskometer level of HDFC Large Cap?",
    "Is there an exit load on HDFC Equity Flexi Cap?",
    "What is the exit load redemption charge on HDFC Flexi Cap?",
    "How do I download my capital gains statement for HDFC Small Cap?",
]


@pytest.mark.parametrize(("rule", "text"), ADVICE_POSITIVES)
def test_advice_detector_fires(rule: str, text: str) -> None:
    assert rule in detect_advice(text), f"{rule} missed: {text}"


@pytest.mark.parametrize("text", ADVICE_NEGATIVES)
def test_advice_detector_stays_silent_on_factual_queries(text: str) -> None:
    assert detect_advice(text) == [], text


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #

PERFORMANCE_POSITIVES: list[str] = [
    "What is the 1 year return of HDFC Large Cap?",
    "Show me the YTD performance.",
    "Which of these five funds performed best last year?",
    "Which scheme outperformed the rest?",
    "Compare returns of ELSS vs Large Cap.",
    "What is the yield of the flexi cap fund?",
    "How much did the small cap fund earn?",
    "growth of 10000 in HDFC Large Cap",
    "cagr of the balanced advantage fund",
    "How did HDFC Large Cap perform last year?",
    "how has the flexi cap fund performed",
    "performance last 3 years",
]

PERFORMANCE_NEGATIVES: list[str] = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "What is the minimum SIP amount of HDFC ELSS Tax Saver?",
    "What benchmark does HDFC Balanced Advantage Fund track?",
    "What is the riskometer risk level?",
    "expense ratio of HDFC Large Cap Direct Growth plan",
    "What is the AUM of HDFC Equity Flexi Cap?",
]


@pytest.mark.parametrize("text", PERFORMANCE_POSITIVES)
def test_performance_detector_fires(text: str) -> None:
    assert detect_performance(text), f"performance missed: {text}"


@pytest.mark.parametrize("text", PERFORMANCE_NEGATIVES)
def test_performance_detector_stays_silent_on_factual_queries(text: str) -> None:
    assert detect_performance(text) == [], text


# --------------------------------------------------------------------------- #
# Scope
# --------------------------------------------------------------------------- #

SCOPE_POSITIVES: list[str] = [
    "SBI Bluechip expense ratio",
    "Axis Large Cap",
    "ICICI Princap expense ratio",
    "Kotak Flexi Cap exit load",
    "Motilal Oswal Midcap minimum SIP",
]

SCOPE_NEGATIVES: list[str] = [
    "expense ratio of HDFC Large Cap Direct Growth",
    "lock-in period of HDFC ELSS Tax Saver",
    "benchmark of HDFC Balanced Advantage",
    "riskometer of HDFC Equity Flexi Cap",
]


@pytest.mark.parametrize("text", SCOPE_POSITIVES)
def test_out_of_scope_amc_is_detected(text: str) -> None:
    assert detect_scope(text), f"out-of-scope AMC missed: {text}"


@pytest.mark.parametrize("text", SCOPE_NEGATIVES)
def test_in_scope_schemes_are_not_flagged_as_other_amcs(text: str) -> None:
    assert detect_scope(text) is None, text


# --------------------------------------------------------------------------- #
# Rule coverage - a pattern with no sample is a policy hole
# --------------------------------------------------------------------------- #


def test_every_expected_rule_has_positive_and_negative_samples() -> None:
    # Every performance positive below exercises the single performance rule, and the
    # scope rule is covered by SCOPE_POSITIVES.
    covered = {rule for rule, _text in PII_POSITIVES}
    covered |= {rule for rule, _text in ADVICE_POSITIVES}
    covered |= {"performance.compare", SCOPE_RULE}
    negatives = (
        PII_NEGATIVES + ADVICE_NEGATIVES + PERFORMANCE_NEGATIVES + SCOPE_NEGATIVES
    )
    missing_positives = [rule for rule in EXPECTED_RULES if rule not in covered]
    assert not missing_positives, f"rules with no positive sample: {missing_positives}"
    assert len(negatives) >= 20, "the negative corpus is too small to be meaningful"
    # Every negative must be silent under all three detectors.
    for text in negatives:
        assert detect_pii(text) == [], text
        assert detect_advice(text) == [], text
        assert detect_performance(text) == [], text


# --------------------------------------------------------------------------- #
# Phase A
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("My PAN is ABCDE1234F, what is the expense ratio?", "pii"),
        ("Should I buy HDFC Small Cap?", "advice"),
        ("which fund performed best last year", "performance"),
        ("SBI Bluechip expense ratio", "out_of_scope"),
    ],
)
def test_phase_a_refuses_with_the_right_kind(raw: str, kind: str) -> None:
    result = check_phase_a(raw, query.build_plan(raw))
    assert result is not None
    assert result.kind == kind


@pytest.mark.parametrize(
    "raw",
    [
        "What is the expense ratio of HDFC Large Cap Direct Growth?",
        "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
        "How do I download my capital gains statement?",
        "What is the riskometer of HDFC Balanced Advantage?",
        "what is the atmospheric pressure on Jupiter",
        "",
    ],
)
def test_phase_a_lets_ordinary_questions_through(raw: str) -> None:
    assert check_phase_a(raw, query.build_plan(raw) if raw else None) is None


def test_pii_wins_over_advice() -> None:
    """Priority: the personal data must be refused before anything else runs."""
    raw = "Is HDFC Small Cap good for me? My PAN is ABCDE1234F"
    assert detect_pii(raw) and detect_advice(raw), "both groups must fire for this test"
    result = check_phase_a(raw, query.build_plan(raw))
    assert result is not None
    assert result.kind == "pii"
    assert result.rule == "pii.pan"


def test_pii_wins_over_performance() -> None:
    raw = "what are my returns? my aadhaar is 1234 5678 9012"
    assert detect_pii(raw) and detect_performance(raw)
    result = check_phase_a(raw, query.build_plan(raw))
    assert result is not None and result.kind == "pii"


def test_advice_wins_over_scope() -> None:
    raw = "Should I buy SBI Bluechip?"
    assert detect_advice(raw) and detect_scope(raw)
    result = check_phase_a(raw, query.build_plan(raw))
    assert result is not None
    assert result.kind == "advice"


def test_phase_a_works_without_a_plan() -> None:
    """The link degrades to the default source; the refusal does not."""
    result = check_phase_a("Should I buy HDFC Small Cap?")
    assert result is not None
    assert result.kind == "advice"
    assert config.normalize_url(result.link) in config.allowed_url_keys()


def test_phase_a_rule_is_recorded_for_the_trace() -> None:
    result = check_phase_a("Should I buy HDFC Small Cap?")
    assert result is not None
    assert result.rule in config.ADVICE_PATTERNS


# --------------------------------------------------------------------------- #
# Phase B
# --------------------------------------------------------------------------- #


def _chunk(label: str, text: str, *, scheme: str = "hdfc-large-cap") -> RetrievedChunk:
    """Build a :class:`RetrievedChunk` with just the fields Phase B reads."""
    return RetrievedChunk(
        chunk={
            "label": label,
            "section": "Section",
            "text": text,
            "scheme_key": scheme,
            "source_url": config.SOURCES[0].url,
            "fetched_at": "2026-09-27",
        },
        score=0.7,
        rank=1,
    )


def _result(chunks: list[RetrievedChunk], confidence: float = 0.7) -> RetrievalResult:
    return RetrievalResult(
        chunks=chunks,
        dropped_low_score=0,
        dropped_scheme_mismatch=0,
        confidence=confidence if chunks else 0.0,
        used_filter=False,
    )


def test_empty_retrieval_is_a_low_confidence_refusal() -> None:
    plan = query.build_plan("who manages the fund for HDFC Large Cap?")
    result = check_phase_b(plan, _result([]))
    assert result is not None
    assert result.kind == "low_confidence"
    assert result.rule == "confidence.low"


def test_below_floor_is_a_low_confidence_refusal() -> None:
    plan = query.build_plan("expense ratio of HDFC Large Cap")
    floor = config.SIMILARITY_FLOOR
    result = check_phase_b(plan, _result([_chunk("Expense ratio", "1.04%")], floor - 0.01))
    assert result is not None
    assert result.kind == "low_confidence"


def test_nav_intent_is_redirected_not_answered() -> None:
    """A static snapshot cannot answer a live-NAV question."""
    plan = query.build_plan("What is the current NAV of HDFC Large Cap?")
    assert plan.intent == "nav"
    result = check_phase_b(
        plan, _result([_chunk("NAV", "NAV: 01 Oct '26\n\u20b91,151.92")])
    )
    assert result is not None
    assert result.kind == "nav"
    assert result.rule == "intent.nav"


def test_ungrounded_intent_is_refused() -> None:
    """Nothing in the retrieved text carries the tax fact the question asked for."""
    plan = query.build_plan("What is the section 80c tax benefit of HDFC ELSS?")
    assert plan.intent == "tax"
    result = check_phase_b(plan, _result([_chunk("Minimum SIP", "Minimum SIP is \u20b9500.")]))
    assert result is not None
    assert result.kind == "unknown_intent"
    assert result.rule == "intent.grounded"


def test_grounded_intent_passes() -> None:
    plan = query.build_plan("What is the minimum SIP of HDFC ELSS Tax Saver?")
    result = check_phase_b(plan, _result([_chunk("Minimum SIP", "Minimum SIP is \u20b9500.")]))
    assert result is None


def test_unknown_intent_is_not_grounding_checked() -> None:
    """Nothing to ground against means no refusal from this rule."""
    plan = query.build_plan("tell me something interesting about this scheme")
    assert plan.intent == "unknown"
    result = check_phase_b(plan, _result([_chunk("Fund management", "Dhruv Muchhal.")]))
    assert result is None


def test_grounding_prefers_the_label_then_the_text() -> None:
    by_label = _result([_chunk("Expense ratio", "1.04%")])
    by_text = _result([_chunk("Key facts", "Expense ratio 1.04%")])
    assert grounded("expense_ratio", by_label)[0] is True
    assert grounded("expense_ratio", by_text)[0] is True
    assert grounded("lock_in", by_label)[0] is False


def test_grounding_ignores_spelling_differences() -> None:
    result = _result([_chunk("Lock in period", "It is three years.")])
    assert grounded("lock_in", result)[0] is True


@pytest.mark.parametrize(
    "intent",
    [intent for intent in config.KNOWN_INTENTS],
)
def test_every_intent_has_expected_labels(intent: str) -> None:
    assert expected_labels(intent), f"{intent} has no grounding labels"


def test_phase_b_passes_for_every_in_scope_eval_query(index_ready: None) -> None:
    """End-to-end: retrieval plus phase B clears all but the NAV redirect."""
    from src.retriever import EVAL_QUERIES, retrieve

    for case in EVAL_QUERIES:
        if case["handled_by"] != "retrieval":
            continue
        plan = query.build_plan(case["q"])
        result = retrieve(plan)
        refusal = check_phase_b(plan, result)
        if plan.intent == "nav":
            assert refusal is not None and refusal.kind == "nav", case["q"]
        else:
            assert refusal is None, f"{case['q']!r} was refused: {refusal}"


# --------------------------------------------------------------------------- #
# Refusal shape - the invariants that hold for every refusal
# --------------------------------------------------------------------------- #

ALL_KINDS = list(config.REFUSAL_TEMPLATES)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_refusal_is_at_most_three_sentences(kind: str) -> None:
    refusal = render_refusal(kind, config.default_link())
    assert config.sentence_count(refusal.message) <= config.MAX_ANSWER_SENTENCES


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_refusal_contains_no_digits(kind: str) -> None:
    """A number in a refusal would be a fact the corpus never gave us."""
    refusal = render_refusal(kind, config.default_link())
    assert not re.search(r"\d", refusal.message), refusal.message


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_refusal_carries_exactly_one_allow_listed_link(kind: str) -> None:
    refusal = render_refusal(kind, config.default_link())
    assert config.normalize_url(refusal.link) in config.allowed_url_keys()
    urls = re.findall(r"https?://\S+", refusal.message)
    assert urls == [], "the link belongs in .link, not in the message"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_educational_link_is_allow_listed_for_every_kind(kind: str) -> None:
    for plan in (None, query.build_plan("expense ratio of HDFC ELSS Tax Saver")):
        link = educational_link(kind, plan)  # type: ignore[arg-type]
        assert config.normalize_url(link) in config.allowed_url_keys(), (kind, link)


def test_educational_link_prefers_the_matched_scheme() -> None:
    plan = query.build_plan("exit load of HDFC Small Cap")
    expected = config.source_spec("hdfc-small-cap")
    assert expected is not None
    assert educational_link("advice", plan) == expected.url


def test_render_refusal_rejects_a_link_that_is_not_allow_listed() -> None:
    with pytest.raises(ValueError):
        render_refusal("advice", "https://groww.in.evil.com/mutual-funds/x")


def test_render_refusal_rejects_a_link_outside_the_base() -> None:
    with pytest.raises(ValueError):
        render_refusal("low_confidence", "https://example.com/hdfc")


def test_render_refusal_rejects_an_unknown_kind() -> None:
    with pytest.raises(KeyError):
        render_refusal("not_a_kind", config.default_link())  # type: ignore[arg-type]


def test_render_refusal_rejects_a_template_over_the_sentence_cap(monkeypatch) -> None:
    monkeypatch.setitem(
        config.REFUSAL_TEMPLATES, "advice", "One. Two. Three. Four."
    )
    with pytest.raises(ValueError):
        render_refusal("advice", config.default_link())


# --------------------------------------------------------------------------- #
# Trace payloads
# --------------------------------------------------------------------------- #


def test_describe_phase_a_lists_every_group_in_order() -> None:
    described = describe_phase_a("Should I buy HDFC Small Cap?")
    assert [entry["rule"] for entry in described] == [
        "pii.*",
        "advice.*",
        "performance.*",
        "scope.*",
    ]
    assert config.GUARD_ORDER == ("pii", "advice", "performance", "scope")


def test_describe_phase_a_records_the_matched_amc() -> None:
    entries = {entry["rule"]: entry for entry in describe_phase_a("SBI Bluechip")}
    assert entries["scope.*"]["hit"] is True
    assert "sbi" in str(entries["scope.*"]["detail"])


def test_describe_phase_b_reports_the_floor_comparison() -> None:
    plan = query.build_plan("expense ratio of HDFC Large Cap")
    described = {entry["rule"]: entry for entry in describe_phase_b(plan, _result([]))}
    assert described["confidence.low"]["hit"] is True
    assert "floor" in str(described["confidence.low"]["detail"])


def test_describe_phase_b_marks_later_rules_unreached_after_a_refusal() -> None:
    """A trace must not claim rules fired that were never evaluated."""
    plan = query.build_plan("What is the current NAV of HDFC Large Cap?")
    empty = {entry["rule"]: entry for entry in describe_phase_b(plan, _result([]))}
    assert empty["confidence.low"]["hit"] is True
    assert empty["intent.nav"]["hit"] is False
    assert empty["intent.grounded"]["hit"] is False
    grounded_result = {
        entry["rule"]: entry
        for entry in describe_phase_b(
            plan, _result([_chunk("NAV", "NAV: 01 Oct '26")])
        )
    }
    assert grounded_result["intent.nav"]["hit"] is True
    assert grounded_result["intent.grounded"]["hit"] is False
    answered = {
        entry["rule"]: entry
        for entry in describe_phase_b(
            query.build_plan("expense ratio of HDFC Large Cap"),
            _result([_chunk("Expense ratio", "Expense ratio 1.04%")]),
        )
    }
    assert all(entry["hit"] is False for entry in answered.values())


# --------------------------------------------------------------------------- #
# Purity - no LLM, no retrieval
# --------------------------------------------------------------------------- #


def test_guardrails_module_is_llm_free() -> None:
    """No import of the generator, an LLM client, or the retrieval stage.

    Checked on the parsed imports rather than by grepping for the words "generator"
    and "groq": the docstrings legitimately discuss both, and a grep would either be
    trivially defeated by a comment or force the documentation to be vague.
    """
    tree = ast.parse(GUARDRAILS_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = ("generator", "groq", "retriever", "langchain", "chromadb", "sentence_transformers")
    offenders = sorted(name for name in imported if any(word in name for word in banned))
    assert offenders == [], f"guardrails.py must not import {offenders}"
    assert {"src", "src.types", "src.query"} <= imported


def test_importing_guardrails_loads_no_index_or_model_stack() -> None:
    """A refusal must be producible with no index, no model and no network."""
    code = (
        "import sys, src.guardrails;"
        "print(sorted(m for m in sys.modules "
        "if m.startswith(('chromadb', 'sentence_transformers', 'langchain'))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(GUARDRAILS_SOURCE.parent.parent),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]", completed.stdout