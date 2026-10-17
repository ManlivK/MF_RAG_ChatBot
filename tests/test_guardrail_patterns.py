"""Phase 0 tests: guardrail regex behaviour.

Phase 5 will test the guardrail *functions* (ordering, templates, links). This
file pins the *patterns* now, because a gap here is a policy hole: a pattern
that compiles but never fires is worse than no pattern at all.

Each pattern is checked against positive samples (must fire) and negative
samples (must stay silent) - especially the factual queries from the PRD
section 12 eval set, which must NOT be mistaken for advice or performance.
"""

from __future__ import annotations

import re

import pytest

from src import config


def fires(patterns: dict[str, str], text: str) -> list[str]:
    """Rule ids that match ``text``, case-insensitively."""
    return [rule for rule, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]


# --------------------------------------------------------------------------- #
# PII - must fire on the six shapes (PRD C2)
# --------------------------------------------------------------------------- #

PII_POSITIVES = [
    ("pii.pan", "My PAN is ABCDE1234F, can you check my SIP?"),
    ("pii.pan", "pan: aacck7391j"),
    ("pii.aadhaar", "My Aadhaar is 1234 5678 9012"),
    ("pii.aadhaar", "aadhaar 123456789012"),
    ("pii.account", "Account number 1234567890"),
    ("pii.email", "email me at manliv@example.com"),
    ("pii.phone", "call me on +91 98765 43210"),
    ("pii.otp", "my otp is 123456"),
    ("pii.otp", "send the one time password"),
]

PII_NEGATIVES = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "What is the minimum SIP for the direct plan?",
    "What is the exit load after 12 months?",
    "Is there an exit load of 1 percent?",
]


@pytest.mark.parametrize("rule,text", PII_POSITIVES)
def test_pii_pattern_fires(rule: str, text: str) -> None:
    assert rule in fires(config.PII_PATTERNS, text), f"{rule} missed: {text}"


@pytest.mark.parametrize("text", PII_NEGATIVES)
def test_pii_pattern_stays_silent_on_factual_queries(text: str) -> None:
    """A false positive here would refuse every normal question."""
    assert fires(config.PII_PATTERNS, text) == [], text


# --------------------------------------------------------------------------- #
# Advice - must fire on opinionated / portfolio questions (PRD C6)
# --------------------------------------------------------------------------- #

ADVICE_POSITIVES = [
    "Should I buy HDFC Small Cap?",
    "Can I invest in the ELSS fund for my child?",
    "Should I sell after 3 years?",
    "Which fund is best for me?",
    "Is now a good time to invest?",
    "Can you suggest a portfolio?",
    "How should I allocate 60% to small cap?",
    "Which scheme is good for a retirement corpus?",
    "Would you recommend the flexi cap fund?",
]

ADVICE_NEGATIVES = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "What is the minimum lump sum for HDFC Small Cap?",
    "What benchmark does HDFC Balanced Advantage track?",
    "What is the riskometer level of HDFC Large Cap?",
    "Is there an exit load on HDFC Flexi Cap?",
]


@pytest.mark.parametrize("text", ADVICE_POSITIVES)
def test_advice_pattern_fires(text: str) -> None:
    assert fires(config.ADVICE_PATTERNS, text), f"advice missed: {text}"


@pytest.mark.parametrize("text", ADVICE_NEGATIVES)
def test_advice_pattern_stays_silent_on_factual_queries(text: str) -> None:
    assert fires(config.ADVICE_PATTERNS, text) == [], text


# --------------------------------------------------------------------------- #
# Performance - must fire on every return/ranking phrasing (PRD C3)
# --------------------------------------------------------------------------- #

PERFORMANCE_POSITIVES = [
    # The literal eval-set query. This one motivated the pattern in Phase 0.
    "Which of these five funds performed best last year?",
    "Which fund performed best?",
    "which fund is best performing",
    "What is the 1 year return of HDFC Large Cap?",
    "Show me the YTD performance.",
    "What are the returns of HDFC ELSS?",
    "What is the yield of the flexi cap fund?",
    "Which scheme outperformed the rest?",
    "Which is the top performer?",
    "Compare returns of ELSS vs Large Cap.",
    "How much did the small cap fund earn?",
    "growth of 10000 in HDFC Large Cap",
]

PERFORMANCE_NEGATIVES = [
    "What is the expense ratio of HDFC Large Cap Direct Growth?",
    "What is the lock-in period in HDFC ELSS Tax Saver Fund?",
    "How do I download my capital gains statement?",
    "What is the minimum SIP amount?",
    "What benchmark does HDFC Balanced Advantage Fund track?",
    "What is the riskometer risk level?",
    "What is the exit load redemption charge?",
    "What is the NAV today?",
    "What is the AUM of HDFC Flexi Cap?",
]


@pytest.mark.parametrize("text", PERFORMANCE_POSITIVES)
def test_performance_pattern_fires(text: str) -> None:
    assert fires(config.PERFORMANCE_PATTERNS, text), f"performance missed: {text}"


@pytest.mark.parametrize("text", PERFORMANCE_NEGATIVES)
def test_performance_pattern_stays_silent_on_factual_queries(text: str) -> None:
    assert fires(config.PERFORMANCE_PATTERNS, text) == [], text


# --------------------------------------------------------------------------- #
# Direct Growth must not read as "growth" (performance)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "expense ratio of HDFC Large Cap Direct Growth plan",
        "exit load for the direct growth option",
        "minimum SIP in the Growth plan",
    ],
)
def test_direct_growth_is_not_a_performance_question(text: str) -> None:
    """"Direct Growth" is a plan name on every one of our five schemes."""
    assert fires(config.PERFORMANCE_PATTERNS, text) == [], text


# --------------------------------------------------------------------------- #
# Shorthand expansion
# --------------------------------------------------------------------------- #

SHORTHAND_POSITIVES = [
    (r"\bER\b", "what is the ER"),
    (r"\bmin sip\b", "min sip amount"),
    (r"\block in\b", "lock in period"),
]


@pytest.mark.parametrize("pattern,text", SHORTHAND_POSITIVES)
def test_shorthand_expansion_fires(pattern: str, text: str) -> None:
    assert re.search(pattern, text, re.IGNORECASE), text


def test_factual_shorthand_does_not_collide_with_guardrails() -> None:
    """Expanding 'min sip' must not accidentally trip an advice rule."""
    text = "min sip amount for hdfc elss"
    assert fires(config.ADVICE_PATTERNS, text) == []
    assert fires(config.PERFORMANCE_PATTERNS, text) == []
    assert fires(config.PII_PATTERNS, text) == []


# --------------------------------------------------------------------------- #
# Priority: a single query can hit several groups
# --------------------------------------------------------------------------- #


def test_a_pii_and_advice_query_hits_both_groups() -> None:
    """Phase A must be able to see both; GUARD_ORDER decides PII wins."""
    text = "Is HDFC Small Cap good for me? My PAN is ABCDE1234F"
    assert "pii.pan" in fires(config.PII_PATTERNS, text)
    assert fires(config.ADVICE_PATTERNS, text)
    assert config.GUARD_ORDER.index("pii") < config.GUARD_ORDER.index("advice")
