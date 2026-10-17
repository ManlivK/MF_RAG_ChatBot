"""Stage 11 - Citation, cap and freshness (ARCHITECTURE.md section 4.9).

This is the last code that touches an answer. Everything downstream renders what
:func:`finalise` produced and may not append free text.

The invariant it defends is PRD C1/C5: **at most one link per answer, and only a link
we allow-listed.** A model that invents a source is a normal failure mode, not an
exotic one, so the URL never comes from the model at all: it comes from chunk
metadata, and any URL that appears in the model's prose is stripped before the body
is measured. The final :func:`assert_invariants` call makes a bypass a crash rather
than a bad demo.

One deviation from ARCHITECTURE section 4.9 step 2 is deliberate. That section allows
any URL under ``https://groww.in/mutual-funds/``; this module accepts only the seven
URLs in :data:`src.config.ALLOWED_URLS` (exact match after normalisation). A
prefix match would accept ``https://groww.in/mutual-funds/anything-the-model-invented``,
which is the exact failure this stage exists to prevent - the allow-list is a fixed
set of seven pages, so exact membership is the tighter and more honest gate.
"""

from __future__ import annotations

import re

from src import config
from src.types import (
    AssistantMessage,
    LLMResponse,
    RefusalResult,
    RetrievedChunk,
    RetrievalResult,
)

__all__ = [
    "InvariantViolation",
    "URL_PATTERN",
    "normalize_url",
    "verify_url",
    "strip_urls",
    "count_urls",
    "resolve_link",
    "format_last_updated",
    "render_message",
    "finalise",
    "finalise_refusal",
    "assert_invariants",
]


class InvariantViolation(AssertionError):
    """A rendered message broke a promise the citation stage makes.

    Carries the failing invariant's name so a failure in the demo is diagnosable
    without reading this file first.
    """

    def __init__(self, invariant: str, detail: str) -> None:
        super().__init__(f"{invariant} violated: {detail}")
        self.invariant = invariant


#: Matches an http(s) URL in free text, including markdown and angle-bracket forms.
URL_PATTERN = re.compile(r"https?://[^\s\]\[()<>\"'`]+")

#: A citation label left stranded at the end of a sentence by :func:`strip_urls`,
#: e.g. "Source: https://..." becoming "Source:".
_DANGLING_LABEL = re.compile(r"\b(?:source|sources|link|reference)\b\s*:?\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# URL verification
# --------------------------------------------------------------------------- #


def normalize_url(url: str) -> str:
    """Canonical form used for allow-list comparison.

    Args:
        url: Any URL, or junk.

    Returns:
        Lower-cased, scheme-less, ``www``-less, query/fragment-less form without a
        trailing slash. Empty string for empty input. Delegates to
        :func:`src.config.normalize_url` so the guardrails apply the identical
        transformation.
    """
    return config.normalize_url(url)


def verify_url(url: str) -> bool:
    """True only when ``url`` is one of the allow-listed source URLs.

    Trailing slashes, ``?utm_source=`` trackers and ``http`` vs ``https`` are
    normalised away first, so a tracking-parameterised copy of a real source still
    verifies while a look-alike host never does.

    Args:
        url: The candidate.

    Returns:
        ``True`` for an allow-listed source, ``False`` for anything else - including
        an unknown Groww slug and a different host that merely contains the
        allow-list base as a substring.
    """
    normalised = normalize_url(url)
    if not normalised:
        return False
    return normalised in config.allowed_url_keys()


def count_urls(text: str) -> int:
    """Number of URLs in a rendered string (the "exactly one link" assertion)."""
    return len(URL_PATTERN.findall(text or ""))


def strip_urls(text: str) -> tuple[str, bool]:
    """Remove any URL a model wrote into its own prose.

    A hallucinated "Source: https://..." inside the answer body would break the
    one-link invariant and is one copy-paste away from a fake citation, so it is
    deleted before the body is rendered. The real citation comes from chunk
    metadata in :func:`resolve_link`.

    Args:
        text: The model's answer.

    Returns:
        ``(cleaned, had_url)``.
    """
    if not text:
        return "", False
    cleaned = URL_PATTERN.sub("", text)
    had_url = bool(URL_PATTERN.search(text))
    if had_url:
        cleaned = _DANGLING_LABEL.sub("", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[:;,]\s*$", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), had_url


# --------------------------------------------------------------------------- #
# Link resolution
# --------------------------------------------------------------------------- #


def _cited_chunk(resp: LLMResponse, result: RetrievalResult) -> RetrievedChunk | None:
    """The chunk the model says it used, falling back to the top-ranked chunk.

    Args:
        resp: The parsed response.
        result: The retrieval outcome.

    Returns:
        A chunk, or ``None`` when nothing was retrieved.
    """
    if not result.chunks:
        return None
    try:
        index = int(resp.source_index) - 1
    except (TypeError, ValueError):
        index = -1  # a missing or non-numeric index falls back to the top chunk
    if 0 <= index < len(result.chunks):
        return result.chunks[index]
    return result.chunks[0]


def resolve_link(resp: LLMResponse, result: RetrievalResult) -> tuple[str, bool]:
    """Resolve the single link an answer may carry.

    The candidate is the ``source_url`` of the block the model cited. It is verified
    against the allow-list; a candidate that fails is **discarded** rather than
    repaired, and the top retrieved chunk's URL is used instead with
    ``link_substituted=True``.

    Args:
        resp: The parsed response.
        result: The retrieval outcome.

    Returns:
        ``(url, link_substituted)``. ``url`` is ``""`` only when nothing was
        retrieved, which cannot happen for an answer because guardrail B refuses
        empty retrieval first.
    """
    cited = _cited_chunk(resp, result)
    if cited is None:
        return "", False
    candidate = cited.source_url
    if verify_url(candidate):
        return candidate, False
    fallback = result.chunks[0].source_url if result.chunks else ""
    return fallback, True


def format_last_updated(fetched_at_iso: str) -> str:
    """``"2026-09-27"`` -> ``"27 Sep 2026"``.

    Args:
        fetched_at_iso: A chunk's ``fetched_at``.

    Returns:
        The display stamp, or ``""`` when the date is missing or unparseable. The
        value always comes from chunk metadata, never from the model.
    """
    return config.format_last_updated(fetched_at_iso)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_message(msg: AssistantMessage) -> str:
    """Compose the exact text block the UI renders.

    Kept here rather than in ``app.py`` so that the invariants and the UI can never
    disagree about what a rendered answer contains.

    Args:
        msg: The final message.

    Returns:
        The answer body, then the link, then the freshness stamp, then the
        disclaimer. A refusal renders its educational link without a "Source:"
        label and without a freshness stamp, because it is not a sourced answer.
    """
    lines = [msg.text]
    if msg.source_url:
        lines.append(
            f"Source: {msg.source_url}" if msg.kind == "answer" else msg.source_url
        )
    if msg.last_updated:
        lines.append(f"{config.LAST_UPDATED_PREFIX}{msg.last_updated}")
    if msg.disclaimer:
        lines.append(msg.disclaimer)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Finalisation
# --------------------------------------------------------------------------- #


def finalise(resp: LLMResponse, result: RetrievalResult) -> AssistantMessage:
    """Turn a parsed response into the renderable answer.

    Steps, in order: strip any URL the model wrote, apply the sentence cap, resolve
    and verify the link, take the freshness date from the cited chunk's metadata,
    then assert the invariants. ``link_substituted`` is set when the model's own link
    was discarded or its prose carried a URL.

    Args:
        resp: The parsed response.
        result: The retrieval outcome.

    Returns:
        The final :class:`~src.types.AssistantMessage`.

    Raises:
        InvariantViolation: if the assembled message breaks an invariant.
    """
    body, had_url = strip_urls(resp.answer)
    if body:
        body, capped = config.truncate_to_sentences(
            body, config.MAX_ANSWER_SENTENCES
        )
    else:
        capped = False

    url, substituted = resolve_link(resp, result)
    cited = _cited_chunk(resp, result)
    stamp = format_last_updated(cited.fetched_at if cited else "")

    message = AssistantMessage(
        kind="answer",
        text=body,
        source_url=url,
        last_updated=stamp,
        disclaimer=config.DISCLAIMER,
        truncated=resp.truncated or capped,
        link_substituted=substituted or had_url,
    )
    assert_invariants(message)
    return message


def finalise_refusal(refusal: RefusalResult) -> AssistantMessage:
    """Turn a refusal into the renderable message.

    A refusal carries no citation badge and no freshness stamp: it is not an answer
    grounded in a chunk, it is a decision not to answer.

    Args:
        refusal: The Phase A or Phase B refusal.

    Returns:
        The final :class:`~src.types.AssistantMessage`.

    Raises:
        InvariantViolation: if the assembled message breaks an invariant.
    """
    message = AssistantMessage(
        kind="refusal",
        text=refusal.message,
        source_url=refusal.link,
        last_updated="",
        disclaimer=config.DISCLAIMER,
    )
    assert_invariants(message)
    return message


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def assert_invariants(msg: AssistantMessage) -> None:
    """Assert every rendering promise, or raise :class:`InvariantViolation`.

    The checks:

    a. exactly one URL in the rendered text;
    b. that URL passes :func:`verify_url`;
    c. the freshness line is present for an answer;
    d. the disclaimer is present;
    e. the body is at most ``MAX_ANSWER_SENTENCES`` sentences;
    f. a refusal body contains no digits.

    Plus one that is not in the architecture list but is the reason the rest matter:
    the answer **body** itself contains no URL, so the only link a user can click is
    the one we verified.

    Args:
        msg: The message to check.

    Raises:
        InvariantViolation: on the first failed invariant.
    """
    rendered = render_message(msg)

    urls = URL_PATTERN.findall(rendered)
    if len(urls) != 1:
        raise InvariantViolation(
            "a.exactly_one_url",
            f"expected 1 URL in the rendered message, found {len(urls)}: {urls}",
        )
    if not verify_url(urls[0]):
        raise InvariantViolation("b.url_allow_listed", f"{urls[0]!r} is not allowed")
    if msg.source_url and normalize_url(urls[0]) != normalize_url(msg.source_url):
        raise InvariantViolation(
            "b.url_matches_message", f"{urls[0]!r} != {msg.source_url!r}"
        )
    if URL_PATTERN.search(msg.text):
        raise InvariantViolation(
            "g.body_has_no_url", f"the answer body contains {msg.text!r}"
        )
    if msg.kind == "answer" and config.LAST_UPDATED_PREFIX not in rendered:
        raise InvariantViolation(
            "c.last_updated_present", "an answer must carry a freshness stamp"
        )
    if msg.kind == "answer" and not msg.last_updated.strip():
        raise InvariantViolation(
            "c.last_updated_value", "last_updated must come from chunk metadata"
        )
    if config.DISCLAIMER not in rendered:
        raise InvariantViolation(
            "d.disclaimer_present", "every message must carry the disclaimer"
        )
    sentences = config.sentence_count(msg.text)
    if sentences > config.MAX_ANSWER_SENTENCES:
        raise InvariantViolation(
            "e.sentence_cap",
            f"{sentences} sentences, cap is {config.MAX_ANSWER_SENTENCES}",
        )
    if msg.kind == "refusal" and re.search(r"\d", msg.text):
        raise InvariantViolation(
            "f.refusal_has_no_digits",
            "a number in a refusal would be a fact the corpus never gave us",
        )