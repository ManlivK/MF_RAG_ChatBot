"""Stage 9 - Guardrails, both phases (ARCHITECTURE.md section 4.7).

Two phases, one output type, and **no LLM anywhere in this file**. Every decision
is a regex match or a comparison against a configured constant, which is what makes
refusal behaviour something you can demonstrate live and test exhaustively rather
than something you have to take on trust.

Phase A runs on the raw question *before* anything is embedded, logged or sent
anywhere, in the fixed order ``pii -> advice -> performance -> scope`` (see
:data:`src.config.GUARD_ORDER`). PII is first and unconditional: when it fires, the
raw text is dropped without ever reaching the retriever, the generator or the audit
log.

Phase B runs on the retrieval outcome: nothing above the similarity floor, a live-NAV
question we cannot answer from a static snapshot, or a recognised intent whose
expected fact label appears nowhere in what was retrieved.

The one deviation from ARCHITECTURE section 4.7 is documented on
:func:`grounded`: the grounding surface includes a retrieved chunk's ``text`` and not
only its ``label``/``section``, because the Phase 2 label extractor attaches the
document title to most chunks ("Very High Risk" lives in a chunk labelled
"Expense ratio"). Grounding on labels alone would refuse every riskometer question.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src import config
from src.query import mentions_other_amc
from src.types import Intent, QueryPlan, RefusalKind, RefusalResult, RetrievalResult

__all__ = [
    "SCOPE_RULE",
    "detect_pii",
    "detect_advice",
    "detect_performance",
    "detect_scope",
    "check_phase_a",
    "check_phase_b",
    "grounded",
    "grounding_evidence",
    "expected_labels",
    "factsheet_link",
    "educational_link",
    "render_refusal",
    "describe_phase_a",
    "describe_phase_b",
]

#: Rule id recorded for the out-of-scope AMC refusal. The matched AMC name is kept in
#: the trace (``describe_phase_a``) rather than in the rule id, so rule ids stay a
#: closed, comparable vocabulary.
SCOPE_RULE: str = "scope.other_amc"

#: Substring of a URL that marks a factsheet link, checked by
#: :func:`_factsheet_link`. Nothing on Groww's scheme pages is published as a
#: downloadable factsheet, so this scan normally finds nothing and the performance
#: refusal falls back to the scheme page - which is recorded here rather than
#: assumed.
FACTSHEET_MARKER: str = "factsheet"


# --------------------------------------------------------------------------- #
# Rule evaluation
# --------------------------------------------------------------------------- #


def _hits(patterns: dict[str, str], text: str) -> list[str]:
    """Rule ids from ``patterns`` that match ``text``.

    Args:
        patterns: A rule -> regex map from ``config``.
        text: The string to test.

    Returns:
        Matching rule ids in declaration order, so the first one is stable and can
        be used as the trace's rule id.
    """
    if not text:
        return []
    return [
        rule
        for rule, pattern in patterns.items()
        if re.search(pattern, text, re.IGNORECASE)
    ]


def detect_pii(text: str) -> list[str]:
    """Detect personal data in a question.

    Args:
        text: The user's raw message.

    Returns:
        The PII rule ids that fired, e.g. ``["pii.pan"]``. An empty list means the
        text is safe to forward, embed and log.
    """
    return _hits(config.PII_PATTERNS, text)


def detect_advice(text: str) -> list[str]:
    """Detect a request for an opinion, a recommendation or a trade instruction.

    Args:
        text: The user's raw message.

    Returns:
        The advice rule ids that fired.
    """
    return _hits(config.ADVICE_PATTERNS, text)


def detect_performance(text: str) -> list[str]:
    """Detect a request for returns, yield, ranking or comparison.

    ``nav`` is deliberately not here: a live-NAV question gets its own redirect from
    Phase B's ``intent.nav`` rule, with a message about the static snapshot rather
    than a message about past performance.

    Args:
        text: The user's raw message.

    Returns:
        The performance rule ids that fired.
    """
    return _hits(config.PERFORMANCE_PATTERNS, text)


def detect_scope(text: str) -> str | None:
    """Detect an AMC that is outside the corpus.

    Args:
        text: The user's raw message.

    Returns:
        The matched AMC name (``"sbi bluechip"``), or ``None`` when every AMC
        mentioned is in scope. Alias matching is delegated to
        :func:`src.query.mentions_other_amc` so the corpus and the guardrail can
        never disagree about which AMCs are out of scope.
    """
    return mentions_other_amc(text or "")


# --------------------------------------------------------------------------- #
# Phase A - pre-retrieval
# --------------------------------------------------------------------------- #


def check_phase_a(raw: str, plan: QueryPlan | None = None) -> RefusalResult | None:
    """Run the pre-retrieval guards, in the fixed order ``pii -> advice ->
    performance -> scope``.

    The order is a safety property, not a preference: PII first means a message that
    both contains a PAN and asks for advice is refused as PII, so the personal data
    never reaches the retrieval step or the log.

    Args:
        raw: The user's message, verbatim. Matching happens on this, not on the
            expanded query, so nothing a rewrite introduced can smuggle a rule past.
        plan: The Stage 5 plan, when it already exists. Used only to pick the most
            helpful educational link; a refusal is still produced without it.

    Returns:
        A :class:`~src.types.RefusalResult`, or ``None`` when the question may
        proceed to retrieval.
    """
    pii = detect_pii(raw)
    if pii:
        return render_refusal("pii", educational_link("pii", plan), pii[0])

    advice = detect_advice(raw)
    if advice:
        return render_refusal("advice", educational_link("advice", plan), advice[0])

    performance = detect_performance(raw)
    if performance:
        return render_refusal(
            "performance", educational_link("performance", plan), performance[0]
        )

    other_amc = detect_scope(raw)
    if other_amc:
        return render_refusal(
            "out_of_scope", educational_link("out_of_scope", plan), SCOPE_RULE
        )

    return None


# --------------------------------------------------------------------------- #
# Phase B - post-retrieval
# --------------------------------------------------------------------------- #


def expected_labels(intent: Intent) -> list[str]:
    """The chunk labels that count as grounding for an intent.

    Args:
        intent: A Stage 5 intent.

    Returns:
        The configured ``expected_labels`` for that intent; empty for
        ``"unknown"``, which is never grounded-checked.
    """
    return list(config.FACT_TYPES.get(intent, {}).get("expected_labels", []))


def _loose(text: str) -> str:
    """Lower-case, hyphen- and space-insensitive form used for label comparison.

    The corpus writes "Lock-in period", "Lock in period" and "lock in" for the same
    thing; comparing on a normalised form keeps the guardrail from depending on which
    spelling the chunker happened to emit.
    """
    return re.sub(r"[\s\-_/]+", " ", text.lower()).strip()


def _contains(haystack: str, needle: str) -> bool:
    """True when ``needle`` appears in ``haystack``, ignoring case and separators."""
    haystack_loose = _loose(haystack)
    needle_loose = _loose(needle)
    return bool(needle_loose) and needle_loose in haystack_loose


def grounding_evidence(
    intent: Intent, result: RetrievalResult
) -> tuple[RetrievedChunk | None, str]:
    """Find the retrieved chunk that carries the asked-for fact, and say why.

    The grounding surface is a chunk's ``label`` + ``section`` first, then its
    ``text``. Labels are checked first because they are what Stage 2 extracted; the
    text fallback exists because most chunks carry the document title as their label
    and keep the real facts in the body ("Very High Risk" sits in the chunk labelled
    "Expense ratio"). Without it every riskometer question would be refused as
    ungrounded, which is exactly the false refusal the similarity floor was supposed
    to prevent.

    Args:
        intent: The Stage 5 intent.
        result: The retrieval outcome.

    Returns:
        ``(chunk, label)`` for the first match, or ``(None, "")`` when nothing
        matched. An intent with no expected labels also returns ``(None, "")`` -
        there is nothing to check - which is why callers must treat an unknown intent
        as "not grounded-checked" rather than "ungrounded".
    """
    labels = expected_labels(intent)
    if not labels:
        return None, ""

    # Labels are the outer loop on purpose: the first expected label wins over a
    # higher-scoring chunk that only matched a later, vaguer label. Without it an
    # "expense ratio" question is answered from the chunk labelled "Exit load",
    # because that chunk outranks the one holding the ratio.
    for label in labels:
        for retrieved in result.chunks:
            if _contains(f"{retrieved.label} {retrieved.section}", label):
                return retrieved, label

    for label in labels:
        for retrieved in result.chunks:
            if _contains(retrieved.text, label):
                return retrieved, label

    return None, ""


def grounded(intent: Intent, result: RetrievalResult) -> tuple[bool, str]:
    """Decide whether the retrieved chunks actually contain the asked-for fact.

    Thin predicate over :func:`grounding_evidence`, which is where the work is done;
    this is the shape :func:`check_phase_b` wants.

    Args:
        intent: The Stage 5 intent.
        result: The retrieval outcome.

    Returns:
        ``(True, evidence)`` where ``evidence`` is the expected label that matched,
        or ``(True, "")`` when the intent has no expected labels (nothing to check
        against) and ``(False, "")`` when nothing matched.
    """
    if not expected_labels(intent):
        return True, ""
    _chunk, label = grounding_evidence(intent, result)
    return (True, label) if label else (False, "")


def check_phase_b(plan: QueryPlan, result: RetrievalResult) -> RefusalResult | None:
    """Run the post-retrieval guards: ``confidence.low`` -> ``intent.nav`` ->
    ``intent.grounded``.

    Args:
        plan: The Stage 5 plan.
        result: The retrieval outcome.

    Returns:
        A :class:`~src.types.RefusalResult`, or ``None`` when the question may go to
        the generator. The floor is read from :mod:`src.config` rather than from the
        retriever, which keeps this module free of any retrieval import.
    """
    floor = config.similarity_floor_for(plan.intent)
    if result.is_empty or result.confidence < floor:
        return render_refusal(
            "low_confidence", educational_link("low_confidence", plan), "confidence.low"
        )

    if plan.intent == "nav":
        return render_refusal("nav", educational_link("nav", plan), "intent.nav")

    is_grounded, _evidence = grounded(plan.intent, result)
    if not is_grounded:
        return render_refusal(
            "unknown_intent",
            educational_link("unknown_intent", plan),
            "intent.grounded",
        )

    return None


# --------------------------------------------------------------------------- #
# Refusal rendering
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def factsheet_link() -> str | None:
    """Find an allow-listed factsheet URL in the committed corpus.

    Cached because it reads files off disk and the corpus is static within a run.

    Returns:
        The URL, or ``None`` when the corpus publishes no factsheet link. Nothing on
        the scraped scheme pages is published as one, so this is normally ``None`` and
        the performance refusal falls back to the scheme page. The scan is strict on
        purpose: a factsheet that is not on the allow-list cannot be linked, so it
        is ignored rather than rendered.
    """
    allowed = config.allowed_url_keys()
    for path in sorted(config.PATHS["CORPUS_DIR"].glob("*.md")):
        try:
            body = path.read_text(encoding=config.CORPUS_ENCODING)
        except OSError:  # pragma: no cover - unreadable corpus file
            continue
        for token in re.findall(r"https?://[^\s\)\]\|]+", body):
            normalised = config.normalize_url(token)
            if FACTSHEET_MARKER in normalised and normalised in allowed:
                return token
    return None


def educational_link(kind: RefusalKind, plan: QueryPlan | None) -> str:
    """Pick the one educational link a refusal is allowed to carry.

    Always an allow-listed URL: the matched scheme's page when Stage 5 resolved a
    scheme, otherwise the first scheme page. The performance refusal prefers a
    factsheet link if the corpus ever publishes one.

    Args:
        kind: The refusal kind. Only ``"performance"`` changes the choice.
        plan: The Stage 5 plan, when available.

    Returns:
        An allow-listed URL. Never empty.
    """
    if kind == "performance":
        published = factsheet_link()
        if published:
            return published
    spec = (
        config.source_spec(plan.scheme_key) if plan and plan.scheme_key else None
    )
    return spec.url if spec else config.default_link()


def _is_allow_listed(url: str) -> bool:
    """True when ``url`` normalises to an entry in the allow-list.

    Mirrors :func:`src.citation.verify_url` without importing it, so
    :mod:`src.guardrails` stays independent of the citation stage.
    """
    return bool(url) and config.normalize_url(url) in config.allowed_url_keys()


def render_refusal(
    kind: RefusalKind, link: str, rule: str = ""
) -> RefusalResult:
    """Render a refusal from its configured template.

    The message is the template body only: no link (that travels in ``link``) and no
    digits, because a number in a refusal would be a fact the corpus never gave us.
    Both properties are enforced here rather than assumed, so a bad template fails
    the moment it is used instead of reaching the UI.

    Args:
        kind: The refusal kind; also the key into
            :data:`src.config.REFUSAL_TEMPLATES`.
        link: The single educational link. Must be allow-listed.
        rule: The rule id for the trace. Defaults to ``"<kind>.*"``.

    Returns:
        The frozen :class:`~src.types.RefusalResult`.

    Raises:
        KeyError: when ``kind`` has no configured template.
        ValueError: when the link is not allow-listed, or the rendered template
            exceeds :data:`src.config.MAX_ANSWER_SENTENCES`.
    """
    template = config.REFUSAL_TEMPLATES[kind]
    count = config.sentence_count(template)
    if count > config.MAX_ANSWER_SENTENCES:
        raise ValueError(
            f"refusal template {kind!r} is {count} sentences, "
            f"the cap is {config.MAX_ANSWER_SENTENCES}"
        )
    if not _is_allow_listed(link):
        raise ValueError(f"refusal {kind!r} carries a non-allow-listed link: {link!r}")
    return RefusalResult(
        kind=kind,
        message=template,
        link=link,
        rule=rule or f"{kind}.*",
    )


# --------------------------------------------------------------------------- #
# Trace payloads (ARCHITECTURE.md section 5.4)
# --------------------------------------------------------------------------- #


def describe_phase_a(text: str) -> list[dict[str, object]]:
    """Summarise every Phase A rule for the UI trace, hit or not.

    A guardrail the user cannot see is a guardrail they cannot trust, so the trace
    lists all four groups with their hit state rather than only the one that fired.

    Args:
        text: The raw question.

    Returns:
        One ``{"rule", "hit", "rules"}`` entry per group, in evaluation order.
    """
    pii = detect_pii(text)
    advice = detect_advice(text)
    performance = detect_performance(text)
    other_amc = detect_scope(text)
    return [
        {"rule": "pii.*", "hit": bool(pii), "rules": pii},
        {"rule": "advice.*", "hit": bool(advice), "rules": advice},
        {"rule": "performance.*", "hit": bool(performance), "rules": performance},
        {
            "rule": "scope.*",
            "hit": bool(other_amc),
            "rules": [SCOPE_RULE],
            "detail": other_amc or "",
        },
    ]


def describe_phase_b(plan: QueryPlan, result: RetrievalResult) -> list[dict[str, object]]:
    """Summarise every Phase B rule for the UI trace, hit or not.

    The rules short-circuit in the same order :func:`check_phase_b` evaluates them, so
    a rule listed as not hit after an earlier one fired is genuinely not hit - it was
    never reached. A trace that claimed every rule fired at once would be a lie the
    demo cannot afford.

    Args:
        plan: The Stage 5 plan.
        result: The retrieval outcome.

    Returns:
        One ``{"rule", "hit", "detail"}`` entry per Phase B rule, in evaluation
        order.
    """
    floor = config.similarity_floor_for(plan.intent)
    confidence_hit = result.is_empty or result.confidence < floor
    reached = not confidence_hit
    nav_hit = reached and plan.intent == "nav"
    reached = reached and not nav_hit
    is_grounded, evidence = grounded(plan.intent, result)
    grounded_hit = reached and not is_grounded
    if grounded_hit:
        grounded_detail = "no expected label in the retrieved chunks"
    elif not reached:
        grounded_detail = "not reached"
    else:
        grounded_detail = f"grounded on {evidence!r}" if evidence else "intent unknown"
    return [
        {
            "rule": "confidence.low",
            "hit": confidence_hit,
            "detail": f"top similarity {result.confidence:.2f} vs floor {floor:.2f}",
        },
        {
            "rule": "intent.nav",
            "hit": nav_hit,
            "detail": f"intent={plan.intent}"
            + ("" if reached else " (not reached)"),
        },
        {"rule": "intent.grounded", "hit": grounded_hit, "detail": grounded_detail},
    ]