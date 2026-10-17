"""Phase 6 - generator tests (IMPLEMENTATION.md section 6.1).

No network and no API key: the model call is the only part that needs one, so it is
the only part left untested here. Everything this file covers - prompt construction,
JSON parsing, index clamping, the sentence cap and the refusal to generate when no key
is set - is deterministic and testable offline, which is exactly what keeps the demo
reproducible on a machine without credentials.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.generator import (
    LLMUnavailable,
    ParseFailure,
    build_prompt,
    enforce_sentence_cap,
    generate,
    llm_available,
    parse_response,
)
from src.types import RetrievedChunk, RetrievalResult


def _result(count: int = 2) -> RetrievalResult:
    spec = config.SOURCES[0]
    chunks = [
        RetrievedChunk(
            chunk={
                "text": "Exit load of 1% if redeemed within 1 year",
                "label": "Exit load",
                "section": "Exit Load",
                "scheme_key": spec.scheme_key,
                "scheme_name": spec.scheme_name,
                "source_url": spec.url,
                "fetched_at": "2026-09-27",
            },
            score=0.7 - index * 0.1,
            rank=index + 1,
        )
        for index in range(count)
    ]
    return RetrievalResult(chunks, 0, 0, 0.7, True)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def test_build_prompt_injects_the_blocks_and_the_query() -> None:
    from src import query
    from src.retriever import format_context

    plan = query.build_plan("exit load of HDFC Large Cap")
    context = format_context(_result())
    prompt = build_prompt(plan, context)
    assert context in prompt
    assert plan.expanded in prompt
    assert config.PROMPT_BLOCKS_TOKEN not in prompt
    assert config.PROMPT_QUERY_TOKEN not in prompt


def test_build_prompt_keeps_the_json_example_intact() -> None:
    from src import query

    prompt = build_prompt(query.build_plan("expense ratio"), "[1] block")
    assert '{"answer": "<1-3 sentences>"' in prompt


def test_prompt_forbids_advice_and_performance() -> None:
    for phrase in ("investment advice", "returns or performance", "At most 3 sentences"):
        assert phrase in config.PROMPT_TEMPLATE


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_parse_response_reads_well_formed_json() -> None:
    raw = json.dumps(
        {"answer": "Exit load is 1%.", "source_index": 2, "insufficient_context": False}
    )
    response = parse_response(raw, n_blocks=5)
    assert response.answer == "Exit load is 1%."
    assert response.source_index == 2
    assert response.insufficient_context is False
    assert response.raw == raw


def test_parse_response_keeps_the_raw_completion() -> None:
    raw = '```json\n{"answer": "Exit load is 1%.", "source_index": 1}\n```'
    assert parse_response(raw, n_blocks=1).raw == raw


def test_parse_response_survives_a_fenced_block() -> None:
    raw = '```json\n{"answer": "Exit load is 1%.", "source_index": 1}\n```'
    assert parse_response(raw, n_blocks=3).answer == "Exit load is 1%."


def test_parse_response_survives_a_leading_comment() -> None:
    raw = 'Here you go: {"answer": "Exit load is 1%.", "source_index": 1} hope that helps'
    assert parse_response(raw, n_blocks=1).answer == "Exit load is 1%."


def test_parse_response_falls_back_to_regex_on_malformed_json() -> None:
    raw = '{"answer": "Exit load is 1%.", "source_index": 2, "insufficient_context": false,}'
    response = parse_response(raw, n_blocks=4)
    assert response.answer == "Exit load is 1%."
    assert response.source_index == 2


def test_parse_response_unescapes_a_regex_salvaged_answer() -> None:
    raw = '{"answer": "The exit load is 1% \\"if redeemed\\" early.", "source_index": 1,}'
    assert 'if redeemed' in parse_response(raw, n_blocks=1).answer


def test_parse_response_reads_insufficient_context() -> None:
    raw = '{"answer": "", "source_index": 1, "insufficient_context": true}'
    response = parse_response(raw, n_blocks=3)
    assert response.insufficient_context is True
    assert response.answer == ""


@pytest.mark.parametrize("flag", ["true", "True", '"true"', "1", "yes"])
def test_insufficient_context_is_read_tolerantly(flag: str) -> None:
    raw = '{"answer": "", "source_index": 1, "insufficient_context": ' + flag + "}"
    assert parse_response(raw, n_blocks=2).insufficient_context is True


@pytest.mark.parametrize("index", [0, -1, 99, "3", None, "abc", 2.9])
def test_source_index_is_clamped_and_never_crashes(index: object) -> None:
    raw = json.dumps({"answer": "Exit load is 1%.", "source_index": index})
    response = parse_response(raw, n_blocks=3)
    assert 1 <= response.source_index <= 3


def test_source_index_is_at_least_one_when_there_are_no_blocks() -> None:
    raw = json.dumps({"answer": "Exit load is 1%.", "source_index": 1})
    assert parse_response(raw, n_blocks=0).source_index == 1


@pytest.mark.parametrize("raw", ["", "not json at all", '{"answer": ""}', "{}"])
def test_unparseable_output_raises_parse_failure(raw: str) -> None:
    with pytest.raises(ParseFailure):
        parse_response(raw, n_blocks=2)


# --------------------------------------------------------------------------- #
# Sentence cap
# --------------------------------------------------------------------------- #


def test_sentence_cap_truncates_to_three() -> None:
    text, truncated = enforce_sentence_cap("One. Two. Three. Four. Five.")
    assert text == "One. Two. Three."
    assert truncated is True


def test_sentence_cap_leaves_a_short_answer_alone() -> None:
    text, truncated = enforce_sentence_cap("Only one.")
    assert (text, truncated) == ("Only one.", False)


def test_sentence_cap_handles_empty_text() -> None:
    assert enforce_sentence_cap("") == ("", False)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def test_generate_refuses_to_run_without_a_key(monkeypatch) -> None:
    """No key must degrade, never raise an opaque client error."""
    monkeypatch.setattr(config, "load_api_key", lambda required=False: None)
    assert llm_available() is False
    with pytest.raises(LLMUnavailable):
        generate(_plan(), _result())


def test_generate_refuses_to_run_in_extractive_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(config, "EXTRACTIVE_ONLY", True)
    assert llm_available() is False
    with pytest.raises(LLMUnavailable):
        generate(_plan(), _result())


def _plan():
    from src import query

    return query.build_plan("exit load of HDFC Large Cap")


def test_call_llm_raises_unavailable_without_a_key(monkeypatch) -> None:
    from src import generator

    monkeypatch.setattr(config, "load_api_key", lambda required=False: None)
    with pytest.raises(LLMUnavailable):
        generator.call_llm("prompt")


def test_call_llm_wraps_client_errors(monkeypatch) -> None:
    """A free-tier 429 must arrive as LLMUnavailable so the pipeline can degrade."""
    from src import generator

    class Boom:
        def invoke(self, prompt: str):  # noqa: ANN001 - stub
            raise RuntimeError("429 rate limited")

    monkeypatch.setattr(config, "load_api_key", lambda required=False: "k")
    monkeypatch.setattr(generator, "_client", lambda: Boom())
    with pytest.raises(LLMUnavailable):
        generator.call_llm("prompt")


def test_call_llm_joins_content_blocks(monkeypatch) -> None:
    from src import generator

    class Message:
        content = [{"type": "text", "text": '{"answer": "ok."}'}]

    class Client:
        def invoke(self, prompt: str) -> Message:  # noqa: ANN001 - stub
            return Message()

    monkeypatch.setattr(config, "load_api_key", lambda required=False: "k")
    monkeypatch.setattr(generator, "_client", lambda: Client())
    assert generator.call_llm("prompt") == '{"answer": "ok."}'