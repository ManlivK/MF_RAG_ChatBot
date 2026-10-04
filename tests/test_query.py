"""Phase 4 - query tests."""
from __future__ import annotations

from src import query


def test_alias_resolution_longest_match():
    assert query.resolve_scheme("large cap fund") == "hdfc-large-cap"
    assert query.resolve_scheme("hdfc balanced advantage") == "hdfc-balanced-advantage"


def test_shorthand_expands():
    exp, aud = query.expand_shorthand("er of elss")
    assert "expense ratio" in exp
    assert "elss tax saver" in exp or "elss" in exp


def test_intent_classification():
    assert query.classify_intent("what is the expense ratio") == "expense_ratio"
    assert query.classify_intent("exit load") == "exit_load"
    assert query.classify_intent("minimum sip amount") == "min_sip"
    assert query.classify_intent("minimum lump sum") == "min_lump_sum"
    assert query.classify_intent("lock in period") == "lock_in"
    assert query.classify_intent("riskometer") == "riskometer"
    assert query.classify_intent("benchmark") == "benchmark"
    assert query.classify_intent("nav") == "nav"
    assert query.classify_intent("how to download capital gains statement") == "statement_download"
    assert query.classify_intent("tax benefit") == "tax"
    assert query.classify_intent("nonsense xyz123") == "unknown"


def test_other_amc_detected():
    assert query.mentions_other_amc("SBI Bluechip expense ratio") is not None