"""Stage 10 - Grounded generation (ARCHITECTURE.md section 4.8).

One call, one job: turn numbered context blocks into at most three grounded
sentences plus the index of the block they came from. There is no conversation
memory, no history and no pleasantry pass, because every extra behaviour here is
another way for a number to enter an answer that is not in the corpus.

Three failure modes are handled explicitly rather than left to chance:

* the model is unreachable or rate-limited -> :class:`LLMUnavailable`;
* the model returns something that is not the requested JSON -> :class:`ParseFailure`;
* the model returns good JSON with a bad ``source_index`` -> clamped, never a crash.

Both exceptions are caught by :mod:`src.pipeline`, which degrades to the extractive
answer. Generation is therefore always optional: the product works with no API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

from src import config
from src.retriever import format_context
from src.types import LLMResponse, QueryPlan, RetrievalResult

__all__ = [
    "LLMUnavailable",
    "ParseFailure",
    "llm_available",
    "build_prompt",
    "call_llm",
    "parse_response",
    "enforce_sentence_cap",
    "generate",
]


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or no key is configured.

    Raised instead of returning an empty answer, so the pipeline can tell
    "degrade to extractive" apart from "the model said it had no context".
    """


class ParseFailure(RuntimeError):
    """The completion could not be read as the requested JSON object."""


#: Fenced ```json blocks some models wrap structured output in.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_INDEX_RE = re.compile(r'"source_index"\s*:\s*(-?\d+)')
_INSUFFICIENT_RE = re.compile(
    r'"insufficient_context"\s*:\s*"?([A-Za-z0-9]+)"?', re.IGNORECASE
)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def llm_available() -> bool:
    """Whether generation should be attempted at all.

    ``EXTRACTIVE_ONLY=1`` forces the offline path (used for the recorded demo and
    for CI), and a missing ``GROQ_API_KEY`` means the same thing. The key value is
    never logged or echoed.

    Returns:
        ``True`` when :func:`generate` is expected to reach the model.
    """
    if config.EXTRACTIVE_ONLY:
        return False
    return config.load_api_key(required=False) is not None


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def build_prompt(plan: QueryPlan, context: str) -> str:
    """Render the configured system prompt with the context blocks substituted.

    The template lives in :data:`src.config.PROMPT_TEMPLATE` so the UI can print it
    verbatim in the pipeline trace, and substitution is done with plain token
    replacement rather than ``str.format`` so the JSON example keeps single braces.

    Args:
        plan: The Stage 5 plan. ``plan.expanded`` is the query the model sees.
        context: The numbered blocks from
            :func:`src.retriever.format_context`.

    Returns:
        The full prompt string.
    """
    return config.render_prompt(config.PROMPT_TEMPLATE, context, plan.expanded)


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #


def _client():
    """Build the ``ChatGroq`` client.

    Imported lazily so that importing this module - which the citation and pipeline
    modules' tests do - never constructs a network client, and so a missing optional
    dependency surfaces as :class:`LLMUnavailable` rather than an import error at
    module load.
    """
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise LLMUnavailable(
            "langchain-groq is not installed. Run: python -m pip install langchain-groq"
        ) from exc

    api_key = config.load_api_key(required=False)
    if not api_key:
        raise LLMUnavailable("GROQ_API_KEY is not set")
    return ChatGroq(
        model=config.GENERATOR_MODEL,
        temperature=config.GENERATOR_TEMPERATURE,
        max_tokens=config.GENERATOR_MAX_TOKENS,
        top_p=config.GENERATOR_TOP_P,
        max_retries=config.GROQ_MAX_RETRIES,
        response_format={"type": "json_object"},
        api_key=api_key,
    )


def call_llm(prompt: str) -> str:
    """Send one prompt and return the raw completion text.

    Retries are handled by the client (``max_retries`` with exponential backoff), so
    a rate-limited free tier does not surface as an exception on the first attempt.

    Args:
        prompt: The rendered prompt.

    Returns:
        The completion exactly as the model produced it, for the trace.

    Raises:
        LLMUnavailable: when the key is missing, the client cannot be built, or the
            request still fails after the configured retries.
    """
    client = _client()
    try:
        message = client.invoke(prompt)
    except Exception as exc:  # noqa: BLE001 - the SDK raises a wide family here
        raise LLMUnavailable(f"Groq call failed after retries: {exc}") from exc
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "").strip()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _payload_from_json(raw: str) -> dict | None:
    """Try to read the completion as a JSON object.

    Handles a bare object, a fenced block, and an object embedded in a sentence.
    """
    candidates = [raw, _FENCE.sub("", raw)]
    candidates.extend(_FENCE.sub("", part) for part in re.findall(r"\{.*\}", raw, re.DOTALL))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _payload_from_regex(raw: str) -> dict | None:
    """Salvage the three fields with regexes when the JSON is malformed.

    Worth having: a truncated or trailing-comma'd object is common enough that
    discarding it would turn a usable answer into a refused one.
    """
    answer = _ANSWER_RE.search(raw)
    index = _INDEX_RE.search(raw)
    insufficient = _INSUFFICIENT_RE.search(raw)
    if not answer and not insufficient:
        return None
    try:
        decoded = json.loads(f'"{answer.group(1)}"') if answer else ""
    except json.JSONDecodeError:
        decoded = answer.group(1) if answer else ""
    return {
        "answer": decoded,
        "source_index": int(index.group(1)) if index else 1,
        "insufficient_context": (
            _coerce_bool(insufficient.group(1)) if insufficient else False
        ),
    }


def _coerce_index(value: object, n_blocks: int) -> int:
    """Clamp a model-supplied block index into ``[1, n_blocks]``.

    A clamped index is strictly better than a crash or a silent 0: the worst case is
    that the citation points at block 1 instead of the block the model meant, and
    every URL still passes the allow-list.
    """
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        index = 1
    upper = max(1, n_blocks)
    return max(1, min(index, upper))


def _coerce_bool(value: object) -> bool:
    """Read ``insufficient_context`` tolerantly (``true``/``"true"``/1)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_response(raw: str, n_blocks: int) -> LLMResponse:
    """Parse, validate and clamp a completion.

    Args:
        raw: The completion text.
        n_blocks: How many context blocks the prompt contained, i.e. the valid range
            for ``source_index``.

    Returns:
        The validated :class:`~src.types.LLMResponse`. ``raw`` is preserved verbatim
        for the trace even when the parse needed the regex fallback.

    Raises:
        ParseFailure: when neither the JSON nor the regex fallback yields an answer,
            and the model did not say it had insufficient context.
    """
    payload = _payload_from_json(raw or "")
    if payload is None:
        payload = _payload_from_regex(raw or "")
    if payload is None:
        raise ParseFailure(f"no answer could be parsed from: {raw!r}")

    answer = str(payload.get("answer") or "").strip()
    insufficient = _coerce_bool(payload.get("insufficient_context"))
    if not answer and not insufficient:
        raise ParseFailure(f"neither answer nor insufficient_context in: {raw!r}")

    return LLMResponse(
        answer="" if insufficient else answer,
        source_index=_coerce_index(payload.get("source_index"), n_blocks),
        insufficient_context=insufficient,
        raw=raw,
    )


def enforce_sentence_cap(answer: str) -> tuple[str, bool]:
    """Cut an answer to :data:`src.config.MAX_ANSWER_SENTENCES` sentences.

    Truncation, not a retry: it is faster, it cannot fail, and the prompt already
    asked for three sentences, so anything longer is the model padding rather than
    withholding an answer.

    Args:
        answer: The model's answer text.

    Returns:
        ``(text, truncated)``.
    """
    return config.truncate_to_sentences(answer, config.MAX_ANSWER_SENTENCES)


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def generate(plan: QueryPlan, result: RetrievalResult) -> LLMResponse:
    """Build the prompt, call the model, parse the reply and apply the sentence cap.

    Args:
        plan: The Stage 5 plan.
        result: The retrieval outcome; its chunks become the numbered context.

    Returns:
        The parsed response with ``answer`` already capped at three sentences.

    Raises:
        LLMUnavailable: when generation is disabled or the call failed.
        ParseFailure: when the completion was unusable.
    """
    if not llm_available():
        raise LLMUnavailable(
            "generation disabled (EXTRACTIVE_ONLY or no GROQ_API_KEY)"
        )
    prompt = build_prompt(plan, format_context(result))
    raw = call_llm(prompt)
    response = parse_response(raw, len(result.chunks))
    body, truncated = enforce_sentence_cap(response.answer)
    return replace(response, answer=body, truncated=response.truncated or truncated)