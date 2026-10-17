"""Phase 9 - unit tests for the eval harness's pure helpers.

The harness itself is exercised end-to-end by
``.\\.venv\\Scripts\\python.exe deliverables\\run_eval.py``. Those tests must not
require the index, the embedding model or an API key, so everything here drives
``run_case`` with a faked ``pipeline.answer`` and tests the assertion helpers in
isolation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src import config
from src.types import AssistantMessage, PipelineResult

ROOT = Path(__file__).resolve().parents[1]

# ``deliverables/`` is not a package, so load the harness by path.
_spec = importlib.util.spec_from_file_location(
    "run_eval", ROOT / "deliverables" / "run_eval.py"
)
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

URL = config.SOURCES[0].url


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #


def _prompt_with_context(context: str) -> str:
    return config.render_prompt(config.PROMPT_TEMPLATE, context, "a query")


def _answer(
    text: str,
    *,
    url: str = URL,
    last_updated: str = "27 Sep 2026",
    context: str = "Expense ratio\n1.04%\n",
) -> PipelineResult:
    message = AssistantMessage(
        kind="answer",
        text=text,
        source_url=url,
        last_updated=last_updated,
        disclaimer=config.DISCLAIMER,
    )
    trace = {"stage_10_generation": {"mode": "extractive", "prompt": _prompt_with_context(context)}}
    return PipelineResult(message, trace)


def _refusal(kind: str = "advice", *, url: str = URL) -> PipelineResult:
    message = AssistantMessage(
        kind="refusal",
        text=config.REFUSAL_TEMPLATES[kind],
        source_url=url,
        last_updated="",
        disclaimer=config.DISCLAIMER,
    )
    return PipelineResult(message, {})


# --------------------------------------------------------------------------- #
# EVAL_QUERIES shape
# --------------------------------------------------------------------------- #


def test_eval_queries_has_the_prd_ten() -> None:
    assert len(run_eval.EVAL_QUERIES) == 10
    for case in run_eval.EVAL_QUERIES:
        assert set(case) >= {"q", "expect", "title"}
        assert case["expect"] in {
            "answer",
            "refusal_advice",
            "refusal_performance",
            "refusal_pii",
            "refusal_nav",
            "refusal_scope",
        }


def test_eval_queries_cover_every_required_refusal() -> None:
    expects = {case["expect"] for case in run_eval.EVAL_QUERIES}
    assert {"answer", "refusal_advice", "refusal_pii", "refusal_nav"} <= expects


def test_eval_queries_use_the_synthetic_pan() -> None:
    pii_case = next(c for c in run_eval.EVAL_QUERIES if c["expect"] == "refusal_pii")
    assert run_eval.PII_SAMPLE in pii_case["q"]


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_check_coerces_to_bool() -> None:
    assert run_eval.check(True, "A1") is True
    assert run_eval.check(False, "A1") is False
    assert run_eval.check("non-empty", "A2") is True
    assert run_eval.check("", "A2") is False


def test_numbers_extracts_digit_tokens() -> None:
    assert run_eval._numbers("Expense ratio 1.04% and AUM 39,933.36 Cr") == [
        "1.04",
        "39,933.36",
    ]
    assert run_eval._numbers("no digits here") == []


def test_context_is_read_from_the_prompt() -> None:
    result = _answer("Expense ratio: 1.04%.")
    context = run_eval._context_from_trace(result.trace)
    assert "1.04%" in context


def test_grounded_accepts_verbatim_numbers() -> None:
    assert run_eval._grounded("Expense ratio: 1.04%.", "Expense ratio\n1.04%\n") is True


def test_grounded_rejects_a_number_absent_from_the_context() -> None:
    assert run_eval._grounded("Expense ratio: 1.05%.", "Expense ratio\n1.04%\n") is False


def test_grounded_rejects_an_empty_body() -> None:
    assert run_eval._grounded("", "Expense ratio\n1.04%\n") is False


def test_grounded_accepts_a_numberless_answer() -> None:
    assert run_eval._grounded("The scheme is open-ended.", "") is True


def test_safe_rejects_performance_phrasing() -> None:
    assert run_eval._safe("The fund returned 12% last year.") is False


def test_safe_accepts_a_plain_fact() -> None:
    assert run_eval._safe("Expense ratio: 1.04%.") is True
    assert run_eval._safe("Minimum Lumpsum Investment is 100.") is True


def test_build_report_mentions_a7_and_every_query() -> None:
    records = [
        {
            "case": run_eval.EVAL_QUERIES[0],
            "message": _answer("Expense ratio: 1.04%.").message,
            "checks": {code: True for code in run_eval._ASSERTION_CODES},
        }
    ]
    report = run_eval._build_report(records, True, "extractive (no model)")
    assert "A7" in report
    assert run_eval.EVAL_QUERIES[0]["title"] in report
    assert "PASS" in report


# --------------------------------------------------------------------------- #
# run_case, driven by a faked pipeline
# --------------------------------------------------------------------------- #


def test_run_case_passes_a_grounded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval.pipeline, "answer", lambda q: _answer("Expense ratio: 1.04%."))
    record = run_eval.run_case(run_eval.EVAL_QUERIES[0], use_llm=False)
    assert record["passed"] is True
    assert all(record["checks"].values())


def test_run_case_fails_a2_on_an_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval.pipeline, "answer", lambda q: _answer(""))
    record = run_eval.run_case(run_eval.EVAL_QUERIES[0], use_llm=False)
    assert record["checks"]["A2"] is False
    assert record["passed"] is False


def test_run_case_fails_a6_on_an_ungrounded_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_eval.pipeline, "answer", lambda q: _answer("Expense ratio: 1.05%.")
    )
    record = run_eval.run_case(run_eval.EVAL_QUERIES[0], use_llm=False)
    assert record["checks"]["A6"] is False


def test_run_case_fails_a1_when_the_body_carries_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_eval.pipeline,
        "answer",
        lambda q: _answer(f"See https://evil.example.com for 1.04%."),
    )
    record = run_eval.run_case(run_eval.EVAL_QUERIES[0], use_llm=False)
    assert record["checks"]["A1"] is False


def test_run_case_passes_a_refusal_case(monkeypatch: pytest.MonkeyPatch) -> None:
    advice_case = next(c for c in run_eval.EVAL_QUERIES if c["expect"] == "refusal_advice")
    monkeypatch.setattr(run_eval.pipeline, "answer", lambda q: _refusal("advice"))
    record = run_eval.run_case(advice_case, use_llm=False)
    assert record["passed"] is True
    assert record["checks"]["A5"] is True


def test_run_case_fails_a5_when_the_wrong_refusal_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    advice_case = next(c for c in run_eval.EVAL_QUERIES if c["expect"] == "refusal_advice")
    monkeypatch.setattr(run_eval.pipeline, "answer", lambda q: _refusal("pii"))
    record = run_eval.run_case(advice_case, use_llm=False)
    assert record["checks"]["A5"] is False


def test_run_case_fails_a1_when_an_answer_was_expected_but_a_refusal_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_eval.pipeline, "answer", lambda q: _refusal("advice"))
    record = run_eval.run_case(run_eval.EVAL_QUERIES[0], use_llm=False)
    assert record["checks"]["A1"] is False


# --------------------------------------------------------------------------- #
# A7 - the PII string never reaches the log
# --------------------------------------------------------------------------- #


def test_the_pii_query_is_redacted_before_it_could_be_written() -> None:
    from src import logger

    assert run_eval.PII_SAMPLE not in logger.redact(f"my PAN is {run_eval.PII_SAMPLE}")
