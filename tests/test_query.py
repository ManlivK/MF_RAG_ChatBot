"""Phase 4 - query rewrite tests (IMPLEMENTATION.md section 4.3).

Stage 5 is deterministic and LLM-free, so these tests assert exact strings and
exact intent labels rather than fuzzy behaviour.
"""

from __future__ import annotations

import pytest

from src import config, query
from src.types import QueryPlan


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("What is the expense ratio?", "what is the expense ratio"),
        ("  Hello   there  ", "there"),
        ("hi, what is the exit load?", "what is the exit load"),
        ("Please tell me the benchmark", "the benchmark"),
        ("Can you tell me the NAV?", "the nav"),
        ("i want to know the AUM?", "the aum"),
        ("exit load!!!", "exit load"),
    ],
)
def test_normalize_strips_case_filler_and_punctuation(raw: str, expected: str) -> None:
    assert query.normalize(raw) == expected


def test_normalize_longest_filler_wins() -> None:
    """"please tell me" must not be eaten as "please", leaving "tell me" behind."""
    assert query.normalize("please tell me the exit load") == "the exit load"


def test_normalize_preserves_the_raw_text() -> None:
    """Guardrails and the audit log must see the untouched message."""
    raw = "Hi, What is the Expense Ratio?"
    plan = query.build_plan(raw)
    assert plan.raw == raw
    assert plan.normalized != raw


def test_normalize_handles_empty_input() -> None:
    assert query.normalize("") == ""


def test_normalize_strips_the_greeting_itself() -> None:
    """"hello there" is itself a filler prefix, so it leaves nothing behind."""
    assert query.normalize("hello there") == ""


# --------------------------------------------------------------------------- #
# expand_shorthand
# --------------------------------------------------------------------------- #


def test_shorthand_er_of_elss() -> None:
    """The case named in IMPLEMENTATION.md section 4.3."""
    expanded, audit = query.expand_shorthand("er of elss")
    assert "expense ratio" in expanded
    assert "elss tax saver" in expanded.lower()
    assert audit, "the audit trail must not be empty when a rule fired"


@pytest.mark.parametrize(
    ("text", "expected_in_expanded"),
    [
        ("min sip", "minimum SIP amount"),
        ("exit load", "redemption charge"),
        ("riskometer", "risk level"),
        ("benchmark", "index"),
        ("aum", "assets under management"),
        ("capital gains", "capital gains statement"),
        ("elss", "ELSS tax saver"),
    ],
)
def test_shorthand_expansions_from_architecture(text: str, expected_in_expanded: str) -> None:
    """Every expansion listed in ARCHITECTURE section 4.5 step 2."""
    expanded, audit = query.expand_shorthand(text)
    assert expected_in_expanded in expanded
    assert audit


def test_shorthand_substitutes_and_does_not_duplicate() -> None:
    """Overlapping rules must not cascade into "minimum SIP amount amount"."""
    expanded, _ = query.expand_shorthand("tell me min sip for hdfc elss tax saver")
    assert "amount amount" not in expanded
    assert "tax saver tax saver" not in expanded
    assert expanded.lower().count("minimum sip amount") == 1


def test_shorthand_is_a_no_op_on_a_fully_spelled_query() -> None:
    """A query that already says "expense ratio" must be left alone."""
    expanded, audit = query.expand_shorthand("what is the expense ratio")
    assert audit == []
    assert "expense ratio" in expanded


def test_shorthand_audit_is_readable() -> None:
    """The audit trail is rendered verbatim in the UI trace, so it must be prose."""
    expanded, audit = query.expand_shorthand("er of elss")
    assert expanded == "expense ratio of ELSS tax saver"
    assert audit == ["er -> expense ratio", "elss -> ELSS tax saver"]


# --------------------------------------------------------------------------- #
# resolve_scheme
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("large cap fund", "hdfc-large-cap"),
        ("hdfc large cap", "hdfc-large-cap"),
        ("hdfc balanced advantage", "hdfc-balanced-advantage"),
        ("flexi cap", "hdfc-equity-flexi-cap"),
        ("small cap", "hdfc-small-cap"),
        ("tax saver fund", "hdfc-elss-tax-saver"),
    ],
)
def test_alias_resolution(text: str, expected: str) -> None:
    assert query.resolve_scheme(text) == expected


def test_alias_resolution_is_longest_match() -> None:
    """The generic scheme must not shadow the specific one (ARCH section 4.5)."""
    assert query.resolve_scheme("hdfc balanced advantage") == "hdfc-balanced-advantage"
    assert query.resolve_scheme("what is the expense ratio of HDFC Large Cap") == (
        "hdfc-large-cap"
    )


def test_alias_resolution_tolerates_optional_brackets() -> None:
    """Users write "HDFC Equity (Flexi Cap) Fund" and "HDFC Equity Flexi Cap Fund"."""
    assert query.resolve_scheme("hdfc equity (flexi cap) fund") == (
        "hdfc-equity-flexi-cap"
    )
    assert query.resolve_scheme("hdfc equity flexi cap fund") == (
        "hdfc-equity-flexi-cap"
    )


def test_alias_resolution_ignores_unrelated_text() -> None:
    assert query.resolve_scheme("what is the atmospheric pressure on jupiter") is None


def test_every_scheme_alias_resolves_to_its_own_scheme() -> None:
    """A typo in one alias list would silently mis-filter retrieval (D4)."""
    for scheme_key, aliases in config.SCHEME_ALIASES.items():
        for alias in aliases:
            assert query.resolve_scheme(alias) == scheme_key, alias


def test_resolve_scheme_does_not_match_inside_a_word() -> None:
    assert query.resolve_scheme("large capitalist") is None


# --------------------------------------------------------------------------- #
# mentions_other_amc
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected_fragments"),
    [
        ("SBI Bluechip expense ratio", ("sbi",)),
        ("Axis Large Cap", ("axis",)),
        ("ICICI Princap", ("icici",)),
        ("Kotak Mahindra Flexi Cap", ("kotak",)),
        ("Nippon India Small Cap", ("nippon",)),
    ],
)
def test_out_of_scope_amc_is_detected(
    text: str, expected_fragments: tuple[str, ...]
) -> None:
    matched = query.mentions_other_amc(text)
    assert matched, f"expected an out-of-scope AMC in {text!r}"
    assert any(fragment in matched for fragment in expected_fragments)


def test_in_scope_hdfc_is_not_an_out_of_scope_amc() -> None:
    assert query.mentions_other_amc("hdfc large cap fund") is None


def test_out_of_scope_amc_returns_the_longest_match() -> None:
    """Longest-match keeps "kotak mahindra" from being reported as "kotak"."""
    assert query.mentions_other_amc("kotak mahindra balanced advantage") == (
        "kotak mahindra"
    )


# --------------------------------------------------------------------------- #
# classify_intent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("what is the expense ratio", "expense_ratio"),
        ("total expense ratio charges", "expense_ratio"),
        ("what is the exit load", "exit_load"),
        ("is there a redemption charge", "exit_load"),
        ("what is the minimum sip amount", "min_sip"),
        ("how do I start a SIP", "min_sip"),
        ("what is the minimum lump sum", "min_lump_sum"),
        ("what is the one time investment minimum", "min_lump_sum"),
        ("what is the lock in period", "lock_in"),
        ("lockin of elss", "lock_in"),
        ("what is the riskometer", "riskometer"),
        ("how risky is this scheme", "riskometer"),
        ("what is the benchmark index", "benchmark"),
        ("what does it track", "benchmark"),
        ("what is the current NAV", "nav"),
        ("what is the aum", "nav"),
        ("how do i download my capital gains statement", "statement_download"),
        ("where can I get my account statement", "statement_download"),
        ("what is the tax benefit", "tax"),
        ("is it eligible for 80c", "tax"),
        ("when was the scheme launched", "inception"),
    ],
)
def test_each_intent_classifies_from_a_paraphrase(text: str, expected: str) -> None:
    """All 11 intents from a paraphrased query, never from the keyword verbatim."""
    assert query.classify_intent(text) == expected


def test_nonsense_query_is_unknown() -> None:
    assert query.classify_intent("nonsense xyz123") == "unknown"
    assert query.classify_intent("what is the atmospheric pressure on jupiter") == (
        "unknown"
    )


def test_a_tie_resolves_to_unknown() -> None:
    """Two intents scoring equally is ambiguous, and ambiguity is a refusal."""
    assert query.classify_intent("riskometer benchmark") == "unknown"


def test_fund_manager_question_is_classified() -> None:
    """The corpus keeps manager bios, so the intent must recognise the phrasing."""
    assert query.classify_intent("who is the fund manager of hdfc large cap") == (
        "fund_manager"
    )
    assert query.classify_intent("who manages this fund") == "fund_manager"


def test_scheme_name_keyword_does_not_steal_the_intent() -> None:
    """A scheme name can carry a generic intent keyword.

    "HDFC ELSS Tax Saver" contains "Tax", which would otherwise make a
    fund-manager question tie with the ``tax`` intent and be refused as
    ambiguous. The multi-word "fund manager" phrase must win.
    """
    assert (
        query.classify_intent(
            "who is the fund manager for hdfc elss tax saver fund direct plan growth"
        )
        == "fund_manager"
    )
    assert query.classify_intent("what is the tax implication of hdfc elss tax saver") == (
        "tax"
    )


def test_every_known_intent_is_reachable() -> None:
    """No intent in FACT_TYPES may be dead code."""
    reachable = {
        query.classify_intent(
            " ".join(spec["keywords"][:2])
        )
        for spec in config.FACT_TYPES.values()
    }
    assert set(config.KNOWN_INTENTS) & reachable, "no intent could be classified"


def test_classified_intent_is_always_in_the_vocabulary() -> None:
    for text in ("expense ratio", "exit load", "benchmark", "totally unrelated words"):
        assert query.classify_intent(text) in (*config.KNOWN_INTENTS, "unknown")


# --------------------------------------------------------------------------- #
# build_plan
# --------------------------------------------------------------------------- #


def test_build_plan_returns_the_documented_fields() -> None:
    plan = query.build_plan("Hi, what is the ER of HDFC ELSS Tax Saver?")
    assert isinstance(plan, QueryPlan)
    assert plan.raw.startswith("Hi,")
    assert plan.normalized == "what is the er of hdfc elss tax saver"
    assert "expense ratio" in plan.expanded
    assert plan.scheme_key == "hdfc-elss-tax-saver"
    assert plan.intent in (*config.KNOWN_INTENTS, "unknown")
    assert isinstance(plan.expansions, list)


def test_build_plan_is_deterministic() -> None:
    """Same string in, same plan out - that is what makes retrieval reproducible."""
    raw = "what is the exit load of HDFC Small Cap Direct Growth"
    assert query.build_plan(raw) == query.build_plan(raw)


def test_build_plan_for_an_out_of_scope_amc_leaves_intent_parseable() -> None:
    """Scope is decided by Phase 5 from the raw text, not smuggled into intent."""
    plan = query.build_plan("SBI Bluechip expense ratio")
    assert plan.scheme_key is None
    assert plan.intent == "expense_ratio"


def test_build_plan_handles_an_empty_query() -> None:
    plan = query.build_plan("")
    assert plan.normalized == ""
    assert plan.scheme_key is None
    assert plan.intent == "unknown"