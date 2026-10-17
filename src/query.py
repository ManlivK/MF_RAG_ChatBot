"""Stage 5 - Query rewrite and scope resolution (ARCHITECTURE.md section 4.5).

Deterministic and LLM-free. Every rule is a regex over ``config``, which keeps
this stage at ~0 ms, makes it exhaustively unit-testable, and removes a whole
class of failure: the same string always produces the same
:class:`~src.types.QueryPlan`, so a bad retrieval is reproducible.
"""

from __future__ import annotations

import re

from src import config
from src.config import FACT_TYPES, FILLER_PREFIXES, OTHER_AMC_ALIASES, SCHEME_ALIASES
from src.types import Intent, QueryPlan

__all__ = [
    "FILLER_PREFIXES",
    "normalize",
    "expand_shorthand",
    "resolve_scheme",
    "mentions_other_amc",
    "classify_intent",
    "build_plan",
]

#: Trailing characters stripped by :func:`normalize`. Question marks and full
#: stops carry no retrieval signal and would otherwise sit inside the embedded
#: string, nudging cosine similarity for no reason.
_TRAILING_PUNCTUATION = re.compile(r"[\s?.!,;:]+$")

#: Parentheses are optional in user phrasing ("HDFC Equity (Flexi Cap) Fund" vs
#: "HDFC Equity Flexi Cap Fund"), so they are dropped on both sides of alias
#: matching before anything is compared.
_BRACKETS = re.compile(r"[()\[\]{}]")


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Compile an alias into a word-boundary regex tolerant of missing brackets.

    Args:
        alias: A scheme or AMC alias from ``config``.

    Returns:
        A case-insensitive pattern matching the alias anywhere in a string.
    """
    cleaned = _BRACKETS.sub(" ", alias.lower())
    parts = [re.escape(part) for part in cleaned.split() if part]
    body = r"[\s\-]+".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def normalize(raw: str) -> str:
    """Lowercase, strip filler and punctuation, collapse whitespace.

    Args:
        raw: The user's message, verbatim.

    Returns:
        The cleaned string. This is what guardrails and the audit log see; the
        original ``raw`` is kept separately on the plan for the trace.
    """
    if not raw:
        return ""
    text = raw.lower().strip()
    # Longest prefix first, so "please tell me" is not eaten as "please" and left
    # behind the dangling "tell me".
    for prefix in sorted(FILLER_PREFIXES, key=len, reverse=True):
        text = re.sub(rf"^{re.escape(prefix)}\b[\s,]*", "", text)
    text = re.sub(r"^[\s?.!,;:]+", "", text)
    text = _TRAILING_PUNCTUATION.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def expand_shorthand(text: str) -> tuple[str, list[str]]:
    """Expand abbreviations and shorthand using ``config.SHORTHAND``.

    Each entry in the map is a regex phrase paired with its replacement, e.g.
    ``\\bER\\b`` -> ``expense ratio``. The replacement is **substituted**, not
    appended, so the embedded string stays the length of a sentence instead of
    doubling with every rule that fires.

    A rule is skipped when its replacement is already present in the text. That
    guard is what stops the overlapping ``min sip`` / ``minimum sip`` rules from
    cascading into each other and emitting "minimum SIP amount amount".

    Args:
        text: A normalized query.

    Returns:
        ``(expanded_text, audit)`` where ``audit`` is a human-readable trail such
        as ``["ER -> expense ratio"]``, rendered in the UI trace.
    """
    expanded = text
    audit: list[str] = []
    for pattern, replacement in config.SHORTHAND.items():
        if _loose(replacement) in _loose(expanded):
            continue
        expanded, count = re.subn(pattern, replacement, expanded, flags=re.IGNORECASE)
        if count:
            audit.append(f"{_phrase_of(pattern)} -> {replacement}")
    return re.sub(r"\s+", " ", expanded).strip(), audit


def _loose(text: str) -> str:
    """Lowercase and collapse hyphens to spaces, for containment checks.

    Lets "lock-in period" be recognised as already present when the query wrote
    "lock in period", so the rule does not emit "lock-in period period".
    """
    return re.sub(r"[\s\-]+", " ", text.lower()).strip()


def _phrase_of(pattern: str) -> str:
    """Render a shorthand regex as the phrase a human recognises.

    Args:
        pattern: A key from ``config.SHORTHAND``.

    Returns:
        The pattern with its regex punctuation stripped, e.g. ``"ER"`` for
        ``"\\bER\\b"`` and ``"exit load"`` for ``"\\bexit\\s+load\\b"``.
    """
    rendered = pattern.replace(r"\b", "").replace(r"\s", " ")
    rendered = re.sub(r"\[[^\]]*\]", "", rendered)
    rendered = re.sub(r"[\\^$*+?.()|{}]", "", rendered)
    return re.sub(r"\s+", " ", rendered).strip()


def resolve_scheme(text: str) -> str | None:
    """Resolve the in-scope scheme mentioned in a query.

    The **longest** alias wins, so ``"hdfc balanced advantage"`` is not shadowed
    by the generic ``"large cap"``-style aliases of another scheme.

    Args:
        text: A normalized or expanded query.

    Returns:
        The canonical ``scheme_key``, or ``None`` when no in-scope scheme is
        named.
    """
    lowered = text.lower()
    best_key: str | None = None
    best_length = 0
    for key, aliases in SCHEME_ALIASES.items():
        for alias in aliases:
            length = len(alias)
            if length <= best_length:
                continue
            if _alias_pattern(alias).search(lowered):
                best_key = key
                best_length = length
    return best_key


def mentions_other_amc(text: str) -> str | None:
    """Detect an out-of-scope AMC name in a query.

    Args:
        text: A normalized or expanded query.

    Returns:
        The matched AMC name, or ``None`` when every AMC mentioned is in scope.
        Phase 5 turns this into the ``scope.other_amc`` refusal; the intent
        vocabulary in ``types.py`` deliberately has no out-of-scope member, so
        scope is not smuggled into ``QueryPlan.intent``.
    """
    lowered = text.lower()
    best_name: str | None = None
    best_length = 0
    for amc in OTHER_AMC_ALIASES:
        length = len(amc)
        if length <= best_length:
            continue
        if _alias_pattern(amc).search(lowered):
            best_name = amc
            best_length = length
    return best_name


def classify_intent(text: str) -> Intent:
    """Score the query against every intent in ``config.FACT_TYPES``.

    A keyword hit scores its own word count, so a specific multi-word phrase
    outranks a generic single word. That matters because a scheme name can
    carry a generic intent keyword: "who is the fund manager of HDFC ELSS Tax
    Saver" hits ``tax`` only through the "Tax" in the scheme's name, and without
    the weighting it would tie with ``fund_manager`` and be refused as
    ambiguous. The highest total wins; a tie resolves to ``"unknown"`` on
    purpose - an ambiguous question is a refusal, not a coin flip (ARCH section
    4.5 step 4).

    Args:
        text: A normalized or expanded query.

    Returns:
        The winning intent, or ``"unknown"`` on a tie or when nothing matched.
    """
    lowered = text.lower()
    scores: dict[str, int] = {}
    for intent, spec in FACT_TYPES.items():
        scores[intent] = sum(
            len(keyword.split())
            for keyword in spec.get("keywords", [])
            if _alias_pattern(keyword).search(lowered)
        )
    best = max(scores.values(), default=0)
    if best == 0:
        return "unknown"
    winners = [intent for intent, score in scores.items() if score == best]
    if len(winners) > 1:
        return "unknown"
    return winners[0]  # type: ignore[return-value]


def build_plan(raw: str) -> QueryPlan:
    """Run the full Stage 5 rewrite: normalize -> expand -> resolve -> classify.

    Args:
        raw: The user's raw message.

    Returns:
        A frozen :class:`~src.types.QueryPlan`. ``expanded`` is what gets
        embedded; the resolved scheme key is folded into it so the metadata
        filter and the query text agree on which scheme is being asked about.
    """
    normalized = normalize(raw)
    expanded, audit = expand_shorthand(normalized)
    scheme_key = resolve_scheme(expanded)
    intent = classify_intent(expanded)
    return QueryPlan(
        raw=raw,
        normalized=normalized,
        expanded=expanded,
        scheme_key=scheme_key,
        intent=intent,
        expansions=audit,
    )